#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_FILES = {
    "FbHdp": SCRIPT_DIR / "FbHdp.txt",
    "WebSearch": SCRIPT_DIR / "WebSearch.txt",
}
COLORS = {
    "FbHdp": "#D87123",
    "WebSearch": "#0072B2",
}


def load_cdf_points(path):
    xs = []
    ys = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit("invalid CDF row in %s at line %d" % (path, lineno))
            x, y = map(float, parts)
            if x <= 0:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise SystemExit("no positive flow sizes found in %s" % path)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def format_size_ticks(value, _position=None):
    if value >= 1e6:
        if value % 1e6 == 0:
            return "%gM" % (value / 1e6)
        return "%.1fM" % (value / 1e6)
    if value >= 1e3:
        if value % 1e3 == 0:
            return "%gK" % (value / 1e3)
        return "%.1fK" % (value / 1e3)
    return "%g" % value


def plot_cdf(series_records, output_path):
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )

    figure, axis = plt.subplots(figsize=(10, 4.8))
    for label, xs, ys, color in series_records:
        axis.plot(xs, ys, linewidth=2.4, label=label, color=color)

    axis.set_xscale("log")
    axis.set_xlim(50, 3e7)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Flow size (bytes)")
    axis.set_ylabel("CDF (%)")
    axis.grid(True, which="both", linestyle="--", alpha=0.32)
    axis.legend(frameon=False, loc="lower right")

    xticks = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    axis.set_xticks(xticks)
    axis.set_xticklabels([format_size_ticks(t) for t in xticks])
    yticks = [0, 20, 40, 60, 80, 100]
    axis.set_yticks(yticks)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot flow-size CDFs for FbHdp and WebSearch.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated figures",
    )
    parser.add_argument(
        "--output-name",
        default="flow_size_cdf",
        help="base filename without extension",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="one or more output formats",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()

    series_records = []
    for label, path in DEFAULT_FILES.items():
        xs, ys = load_cdf_points(path)
        series_records.append((label, xs, ys, COLORS[label]))

    saved_paths = []
    for ext in args.format:
        output_path = output_dir / ("%s.%s" % (args.output_name, ext))
        plot_cdf(series_records, output_path)
        saved_paths.append(output_path)

    for path in saved_paths:
        print("plot_path = %s" % path)


if __name__ == "__main__":
    main()
