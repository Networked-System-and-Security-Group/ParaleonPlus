#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MIX_DIR = ROOT_DIR / "mix"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def parse_experiment_ids(raw_tokens):
    experiment_ids = []
    seen = set()
    for raw_token in raw_tokens:
        for part in raw_token.split(","):
            token = part.strip()
            if not token:
                continue
            experiment_id = int(token)
            if experiment_id in seen:
                continue
            seen.add(experiment_id)
            experiment_ids.append(experiment_id)
    if not experiment_ids:
        raise SystemExit("no experiment ids were provided")
    return experiment_ids


def percentile(sorted_values, ratio):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * ratio
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value * (1.0 - weight) + upper_value * weight


def average(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def human_bytes(size_bytes):
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return "%d%s" % (int(value), units[unit_index])
    if value >= 100:
        return "%.0f%s" % (value, units[unit_index])
    if value >= 10:
        return "%.1f%s" % (value, units[unit_index])
    return "%.2f%s" % (value, units[unit_index])


def format_metric(value):
    if value is None:
        return "NA"
    return "%.6f" % value


def resolve_run_dirs(experiment_ids):
    indexed_dirs = {}
    for path in MIX_DIR.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("-", 1)[0]
        if prefix.isdigit():
            indexed_dirs[int(prefix)] = path
    missing_ids = [str(experiment_id) for experiment_id in experiment_ids if experiment_id not in indexed_dirs]
    if missing_ids:
        raise SystemExit("missing experiment ids: %s" % ", ".join(missing_ids))
    return [indexed_dirs[experiment_id] for experiment_id in experiment_ids]


def load_manifest(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "id": int(run_dir.name.split("-", 1)[0]),
            "files": {},
            "config": {},
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_fct_file(experiment_id, run_dir, manifest):
    manifest_candidates = []
    files_section = manifest.get("files", {})
    config_section = manifest.get("config", {})
    if files_section.get("fct_output"):
        manifest_candidates.append(Path(files_section["fct_output"]))
    if config_section.get("fct_output"):
        manifest_candidates.append(Path(config_section["fct_output"]))
    for candidate in manifest_candidates:
        if candidate.exists():
            return candidate
    if manifest_candidates:
        raise SystemExit(
            "experiment %s FCT file does not exist: %s" % (experiment_id, manifest_candidates[0])
        )

    fallback_candidates = sorted(run_dir.glob("fct*.txt"))
    if fallback_candidates:
        return fallback_candidates[0]
    raise SystemExit("experiment %s has no FCT file under %s" % (experiment_id, run_dir))


def load_normalized_fct_rows(fct_path):
    rows = []
    with fct_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            try:
                size_bytes = int(parts[4])
                fct_ns = int(parts[6])
                standalone_fct_ns = int(parts[7])
            except ValueError as exc:
                raise SystemExit("invalid numeric value in %s:%d" % (fct_path, line_number)) from exc
            if standalone_fct_ns <= 0:
                continue
            rows.append((size_bytes, fct_ns / float(standalone_fct_ns)))
    if not rows:
        raise SystemExit("empty or invalid FCT file: %s" % fct_path)
    return rows


def bucket_index_for_size(size_bytes, upper_bounds):
    index = bisect.bisect_left(upper_bounds, size_bytes)
    return min(index, len(upper_bounds) - 1)


def build_quantile_upper_bounds(sorted_sizes, bucket_count):
    total = len(sorted_sizes)
    upper_bounds = []
    for bucket_index in range(bucket_count):
        position = int(math.ceil((bucket_index + 1) * total / float(bucket_count)) - 1)
        upper_bounds.append(sorted_sizes[min(position, total - 1)])
    return upper_bounds


def build_log_upper_bounds(sorted_sizes, bucket_count):
    min_size = sorted_sizes[0]
    max_size = sorted_sizes[-1]
    if min_size == max_size:
        return [float(max_size)] * bucket_count
    log_min = math.log(float(min_size))
    log_max = math.log(float(max_size))
    upper_bounds = []
    for bucket_index in range(bucket_count):
        if bucket_index == bucket_count - 1:
            upper_bounds.append(float(max_size))
            continue
        upper_bound = math.exp(log_min + (log_max - log_min) * (bucket_index + 1) / float(bucket_count))
        upper_bounds.append(upper_bound)
    return upper_bounds


def build_bucket_plan(all_sizes, bucket_count, bucket_mode):
    if bucket_count <= 0:
        raise SystemExit("bucket count must be positive")
    sorted_sizes = sorted(all_sizes)
    if not sorted_sizes:
        raise SystemExit("no flow sizes found")

    if bucket_mode == "quantile":
        upper_bounds = build_quantile_upper_bounds(sorted_sizes, bucket_count)
    elif bucket_mode == "log":
        upper_bounds = build_log_upper_bounds(sorted_sizes, bucket_count)
    else:
        raise SystemExit("unsupported bucket mode: %s" % bucket_mode)

    bucket_sizes = [[] for _ in range(bucket_count)]
    for size_bytes in sorted_sizes:
        bucket_sizes[bucket_index_for_size(size_bytes, upper_bounds)].append(size_bytes)

    bucket_plan = []
    for bucket_index, sizes in enumerate(bucket_sizes):
        level = "L%02d" % (bucket_index + 1)
        if sizes:
            min_size = sizes[0]
            max_size = sizes[-1]
        else:
            min_size = None
            max_size = None
        if min_size is None:
            size_range = "empty"
        elif min_size == max_size:
            size_range = human_bytes(min_size)
        else:
            size_range = "%s" % (human_bytes(max_size))
        bucket_plan.append(
            {
                "index": bucket_index,
                "level": level,
                "size_range": size_range,
                "min_size_bytes": min_size,
                "max_size_bytes": max_size,
            }
        )
    return bucket_plan, upper_bounds


def summarize_experiment(experiment_id, run_dir):
    manifest = load_manifest(run_dir)
    fct_path = find_fct_file(experiment_id, run_dir, manifest)
    rows = load_normalized_fct_rows(fct_path)
    return {
        "id": experiment_id,
        "run_dir": run_dir,
        "manifest": manifest,
        "fct_path": fct_path,
        "rows": rows,
    }


def compute_bucket_metrics(rows, upper_bounds):
    grouped_values = [[] for _ in range(len(upper_bounds))]
    for size_bytes, normalized_fct in rows:
        grouped_values[bucket_index_for_size(size_bytes, upper_bounds)].append(normalized_fct)
    avg_values = []
    p99_values = []
    for values in grouped_values:
        sorted_values = sorted(values)
        avg_values.append(average(sorted_values))
        p99_values.append(percentile(sorted_values, 0.99))
    return avg_values, p99_values


def build_table_rows(bucket_plan, experiments, metric_key):
    rows = []
    for bucket in bucket_plan:
        row = {
            "level": bucket["level"],
            "size_range": bucket["size_range"],
        }
        for experiment in experiments:
            metric_values = experiment[metric_key]
            row[str(experiment["id"])] = format_metric(metric_values[bucket["index"]])
        rows.append(row)
    return rows


def write_csv_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_table(header, rows):
    lines = [
        "| %s |" % " | ".join(header),
        "| %s |" % " | ".join(["---"] * len(header)),
    ]
    for row in rows:
        lines.append("| %s |" % " | ".join(str(row[column]) for column in header))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute AVG and P99 normalized FCT tables by flow-size bucket for experiment ids."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="experiment ids, for example: --ids 6 7 8 or --ids 6,7,8",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=10,
        help="number of flow-size buckets, default: 10",
    )
    parser.add_argument(
        "--bucket-mode",
        choices=["quantile", "log"],
        default="quantile",
        help="bucket strategy, default: quantile",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated CSV and metadata files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_ids = parse_experiment_ids(args.ids)
    run_dirs = resolve_run_dirs(experiment_ids)

    experiments = []
    all_sizes = []
    for experiment_id, run_dir in zip(experiment_ids, run_dirs):
        experiment = summarize_experiment(experiment_id, run_dir)
        experiments.append(experiment)
        all_sizes.extend(size_bytes for size_bytes, _ in experiment["rows"])

    bucket_plan, upper_bounds = build_bucket_plan(all_sizes, args.bucket_count, args.bucket_mode)
    for experiment in experiments:
        avg_values, p99_values = compute_bucket_metrics(experiment["rows"], upper_bounds)
        experiment["avg_values"] = avg_values
        experiment["p99_values"] = p99_values

    # Build simple DataFrames and print transposed tables (experiments as rows)
    import pandas as pd

    sizes = [b["size_range"] for b in bucket_plan]

    df_avg = pd.DataFrame({"size_range": sizes})
    df_p99 = pd.DataFrame({"size_range": sizes})
    for experiment in experiments:
        df_avg[str(experiment["id"])] = experiment["avg_values"]
        df_p99[str(experiment["id"])] = experiment["p99_values"]

    df_avg = df_avg.set_index("size_range").T
    df_p99 = df_p99.set_index("size_range").T

    print("AVG Normalized FCT (transposed):")
    print(df_avg.to_string(float_format=lambda x: "NA" if pd.isna(x) else "%.2f" % x))
    print()
    print("P99 Normalized FCT (transposed):")
    print(df_p99.to_string(float_format=lambda x: "NA" if pd.isna(x) else "%.2f" % x))


if __name__ == "__main__":
    main()