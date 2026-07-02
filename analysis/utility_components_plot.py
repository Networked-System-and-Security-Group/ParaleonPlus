#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MIX_DIR = ROOT_DIR / "mix"


def parse_scalar(value):
    token = value.strip()
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    try:
        if any(char in token for char in ".eE"):
            return float(token)
        return int(token)
    except ValueError:
        return token


def resolve_run_dir(experiment_id):
    for path in MIX_DIR.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("-", 1)[0]
        if prefix.isdigit() and int(prefix) == experiment_id:
            return path
    raise SystemExit("missing experiment id: %s" % experiment_id)


def load_manifest(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_tuner_log_path(experiment_id):
    run_dir = resolve_run_dir(experiment_id)
    manifest = load_manifest(run_dir)
    tuner_log = manifest.get("files", {}).get("tuner_log") or manifest.get("files", {}).get("controller_log")
    if tuner_log and Path(tuner_log).exists():
        return Path(tuner_log)
    fallback = run_dir / "tuner.log"
    if fallback.exists():
        return fallback
    raise SystemExit("no tuner log found for experiment %s" % experiment_id)


def split_top_level_commas(text):
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


UTILITY_SEGMENT_RE = re.compile(
    r"^(?P<name>\w+)=(?P<norm>[^\s]+) \(raw=(?P<raw>[^,]+), weighted=(?P<weighted>[^)]+)\)$"
)


def parse_utility_line(text):
    values = {}
    for part in split_top_level_commas(text):
        match = UTILITY_SEGMENT_RE.match(part)
        if match:
            name = match.group("name")
            values[name] = parse_scalar(match.group("norm"))
            prefix = name.replace("_norm", "") if name.endswith("_norm") else name
            values[prefix + "_raw"] = parse_scalar(match.group("raw"))
            values[prefix + "_weighted"] = parse_scalar(match.group("weighted"))
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            values[key.strip()] = parse_scalar(value)
    return values


def parse_block(block_text):
    lines = [line.rstrip() for line in block_text.splitlines() if line.strip()]
    if not lines or "=" not in lines[0]:
        return None
    block = {}
    first_key, first_value = lines[0].split("=", 1)
    block[first_key.strip()] = parse_scalar(first_value)
    for line in lines[1:]:
        stripped = line.strip()
        if ":" not in stripped:
            continue
        prefix, content = stripped.split(":", 1)
        content = content.strip()
        if prefix == "utility":
            block["utility"] = parse_utility_line(content)
    return block


def load_blocks(log_path):
    blocks = []
    for raw_block in log_path.read_text(encoding="utf-8").split("\n\n"):
        block = parse_block(raw_block)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise SystemExit("no tuning blocks found in %s" % log_path)
    return blocks


def build_dataframe(blocks):
    rows = []
    for block in blocks:
        utility = dict(block.get("utility", {}))
        rows.append(
            {
                "sim_time": block.get("sim_time"),
                "throughput_norm": utility.get("throughput_norm"),
                "rtt_norm": utility.get("rtt_norm"),
                "pfc_norm": utility.get("pfc_norm"),
                "throughput_raw": utility.get("throughput_raw"),
                "rtt_raw": utility.get("rtt_raw"),
                "pfc_raw": utility.get("pfc_raw"),
                "total": utility.get("total"),
            }
        )
    df = pd.DataFrame(rows)
    df["sim_time"] = pd.to_numeric(df["sim_time"], errors="coerce")
    for column in ["throughput_norm", "rtt_norm", "pfc_norm", "throughput_raw", "rtt_raw", "pfc_raw", "total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["sim_time"])


def main():
    parser = argparse.ArgumentParser(description="Plot throughput/RTT/PFC/total utility history from tuner.log")
    parser.add_argument("--id", type=int, required=True, help="experiment id under mix/")
    parser.add_argument("--output-prefix", default=None, help="optional output prefix under analysis/")
    parser.add_argument("--raw", action="store_true", help="plot raw throughput/rtt/pfc instead of normalized values")
    args = parser.parse_args()

    log_path = resolve_tuner_log_path(args.id)
    df = build_dataframe(load_blocks(log_path))
    prefix = args.output_prefix or ("utility_components_exp_%s" % args.id)
    png_path = SCRIPT_DIR / (prefix + ".png")
    csv_path = SCRIPT_DIR / (prefix + ".csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    if args.raw:
        throughput_col, rtt_col, pfc_col = "throughput_raw", "rtt_raw", "pfc_raw"
        title_suffix = "raw"
    else:
        throughput_col, rtt_col, pfc_col = "throughput_norm", "rtt_norm", "pfc_norm"
        title_suffix = "normalized"
    plt.plot(df["sim_time"], df[throughput_col], label=throughput_col, linewidth=1.6)
    plt.plot(df["sim_time"], df[rtt_col], label=rtt_col, linewidth=1.6)
    plt.plot(df["sim_time"], df[pfc_col], label=pfc_col, linewidth=1.6)
    plt.plot(df["sim_time"], df["total"], label="total", linewidth=1.8)
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Value")
    plt.title("Utility component history (%s, exp %s)" % (title_suffix, args.id))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print("csv = %s" % csv_path)
    print("plot = %s" % png_path)


if __name__ == "__main__":
    main()
