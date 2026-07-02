#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MIX_DIR = ROOT_DIR / "mix"


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


def resolve_run_dir(experiment_id):
    matches = sorted(glob.glob(str(MIX_DIR / f"{experiment_id}-*")))
    if not matches:
        raise SystemExit("missing experiment id: %s" % experiment_id)
    return Path(matches[-1])


def load_manifest(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_fct_path(run_dir, manifest):
    files = manifest.get("files", {})
    config = manifest.get("config", {})
    for candidate in [files.get("fct_output"), config.get("fct_output")]:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    fallback = sorted(run_dir.glob("fct*.txt"))
    if fallback:
        return fallback[0]
    raise SystemExit("no fct file found under %s" % run_dir)


def load_nfct_dataframe(fct_path):
    df = pd.read_csv(
        fct_path,
        sep=" ",
        header=None,
        names=["sip", "dip", "sport", "dport", "size", "start", "fct", "standalone"],
    )
    df["nfct"] = df["fct"] / df["standalone"]
    return df


def compute_point(experiment_id, threshold_bytes):
    run_dir = resolve_run_dir(experiment_id)
    manifest = load_manifest(run_dir)
    label = manifest.get("message", run_dir.name)
    fct_path = resolve_fct_path(run_dir, manifest)
    df = load_nfct_dataframe(fct_path)
    small = df[df["size"] < threshold_bytes]["nfct"]
    large = df[df["size"] >= threshold_bytes]["nfct"]
    return {
        "id": experiment_id,
        "label": label,
        "small_avg_nfct": small.mean(),
        "large_avg_nfct": large.mean(),
    }


def pareto_frontier(df):
    frontier_indices = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            better_or_equal = (
                other["small_avg_nfct"] <= row["small_avg_nfct"]
                and other["large_avg_nfct"] <= row["large_avg_nfct"]
            )
            strictly_better = (
                other["small_avg_nfct"] < row["small_avg_nfct"]
                or other["large_avg_nfct"] < row["large_avg_nfct"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier_indices.append(i)
    return (
        df.loc[frontier_indices]
        .sort_values(["small_avg_nfct", "large_avg_nfct"])
        .reset_index(drop=True)
    )


def build_output_paths(output_prefix):
    base = SCRIPT_DIR / output_prefix
    return {
        "plot": base.with_suffix(".png"),
        "csv": base.with_name(base.name + "_frontier.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontier for experiments using an NFCT threshold split")
    parser.add_argument("--ids", nargs="+", required=True, help="experiment ids or comma-separated experiment ids")
    parser.add_argument("--threshold-kb", type=float, default=119.0, help="threshold in KB, default 119")
    parser.add_argument("--title", default=None, help="optional plot title")
    parser.add_argument("--output-prefix", default="pareto_frontier", help="output prefix under analysis/")
    args = parser.parse_args()

    experiment_ids = parse_experiment_ids(args.ids)
    threshold_bytes = int(round(args.threshold_kb * 1024))

    points = [compute_point(experiment_id, threshold_bytes) for experiment_id in experiment_ids]
    df = pd.DataFrame(points).sort_values("id").reset_index(drop=True)
    frontier = pareto_frontier(df)

    outputs = build_output_paths(args.output_prefix)
    frontier.to_csv(outputs["csv"], index=False)

    plt.figure(figsize=(10, 7))
    plt.scatter(df["small_avg_nfct"], df["large_avg_nfct"], s=65, alpha=0.85, color="#1f77b4")
    plt.plot(frontier["small_avg_nfct"], frontier["large_avg_nfct"], linestyle="--", linewidth=1.8, color="black")
    for _, row in df.iterrows():
        plt.text(row["small_avg_nfct"] + 0.01, row["large_avg_nfct"] + 0.1, str(int(row["id"])), fontsize=8)
    plt.xlabel("Avg NFCT (< threshold)")
    plt.ylabel("Avg NFCT (>= threshold)")
    plt.title(args.title or ("Pareto Frontier (threshold = %.3gKB)" % args.threshold_kb))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outputs["plot"], dpi=180)
    plt.close()

    print("plot = %s" % outputs["plot"])
    print("frontier_csv = %s" % outputs["csv"])
    print()
    print(frontier.to_string(index=False))


if __name__ == "__main__":
    main()
