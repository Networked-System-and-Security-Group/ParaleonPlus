#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MIX_DIR = ROOT_DIR / "mix"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
COLORS = {
    "DCQCN": "#D87123",
    "Expert": "#8C510A",
    "DCQCN+": "#0072B2",
    "Paraleon": "#A14697",
    "ACC": "#69CC00",
}


def parse_experiment_ids(raw_tokens):
    experiment_ids = []
    seen = set()
    for raw_token in raw_tokens:
        for part in str(raw_token).split(","):
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
            "message": run_dir.name,
            "files": {},
            "config": {},
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_fct_file(run_dir, manifest):
    configured_path = manifest.get("files", {}).get("fct_output")
    if configured_path:
        configured = Path(configured_path)
        if configured.exists():
            return configured
    candidates = sorted(run_dir.glob("fct_*.txt"))
    if candidates:
        return candidates[0]
    raise SystemExit("missing FCT file under %s" % run_dir)


def load_fct_ms(fct_path):
    values_ms = []
    with fct_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            values_ms.append(int(parts[6]) / 1_000_000.0)
    if not values_ms:
        raise SystemExit("empty FCT file: %s" % fct_path)
    return np.array(sorted(values_ms), dtype=float)


def build_label(experiment_id, manifest):
    message = manifest.get("message", "")
    if not message:
        return "exp %s" % experiment_id
    return "exp %s (%s)" % (experiment_id, message)


def build_scheme_label(manifest):
    if manifest.get("scheme") == "expert":
        return "Expert"
    message = manifest.get("message", "").lower()
    if "dcqcnplus" in message:
        return "DCQCN+"
    if "paraleonplus" in message or "paraleon" in message:
        return "Paraleon"
    if "acc" in message:
        return "ACC"
    return "DCQCN"


def collapse_experiment_ids(experiment_ids):
    if not experiment_ids:
        return "exp_none"
    return "exp_" + "_".join(str(experiment_id) for experiment_id in experiment_ids)


def plot_cdf(series_records, output_path, title):
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    for _, label, values_ms, color in series_records:
        y = np.arange(1, len(values_ms) + 1, dtype=float) / float(len(values_ms))
        axis.plot(values_ms, y, linewidth=2.2, label=label, color=color)

    axis.set_xlabel("FCT (ms)")
    axis.set_ylabel("CDF")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.0)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FCT CDF curves in milliseconds for experiment ids.")
    parser.add_argument("--ids", nargs="+", required=True, help="experiment ids, e.g. --ids 210 211 212 213")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="directory for the output figure")
    parser.add_argument("--output-name", default=None, help="optional output filename")
    parser.add_argument("--title", default=None, help="optional figure title")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="output image format")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_ids = parse_experiment_ids(args.ids)
    run_dirs = resolve_run_dirs(experiment_ids)

    series_records = []
    for experiment_id, run_dir in zip(experiment_ids, run_dirs):
        manifest = load_manifest(run_dir)
        fct_path = find_fct_file(run_dir, manifest)
        values_ms = load_fct_ms(fct_path)
        scheme_label = build_scheme_label(manifest)
        color = COLORS.get(scheme_label, None)
        series_records.append((experiment_id, scheme_label, values_ms, color))

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "fct_cdf_%s" % collapse_experiment_ids(experiment_ids)
    output_path = output_dir / (args.output_name or ("%s.%s" % (stem, args.format)))
    title = args.title or "FCT CDF (%s)" % ", ".join(str(experiment_id) for experiment_id in experiment_ids)

    plot_cdf(series_records, output_path, title)
    print("plot_path = %s" % output_path)


if __name__ == "__main__":
    main()
