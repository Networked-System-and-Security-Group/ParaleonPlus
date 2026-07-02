#!/usr/bin/env python3

import argparse
import glob
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


def resolve_tuner_log(experiment_id):
    run_dir = resolve_run_dir(experiment_id)
    manifest = load_manifest(run_dir)
    tuner_log = manifest.get("files", {}).get("tuner_log") or manifest.get("files", {}).get("controller_log")
    if tuner_log and Path(tuner_log).exists():
        return Path(tuner_log)
    fallback = run_dir / "tuner.log"
    if fallback.exists():
        return fallback
    raise SystemExit("missing tuner.log for experiment %s" % experiment_id)


def parse_round_records(log_path):
    blocks = [b for b in log_path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    current_value = None
    rows = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("sim_time="):
            continue

        sim_time = float(lines[0].split("=", 1)[1])
        utility_line = next((ln for ln in lines if ln.startswith("utility:")), None)
        tuning_line = next((ln for ln in lines if ln.startswith("tuning:")), None)
        if utility_line is None or tuning_line is None:
            continue

        total_m = re.search(r"total=([0-9.]+)", utility_line)
        round_m = re.search(r"round=([0-9]+)", tuning_line)
        state_m = re.search(r"state=([a-z_]+)", tuning_line)
        acc_m = re.search(r"accepted=(true|false)", tuning_line)
        if not (total_m and round_m and state_m):
            continue

        total = float(total_m.group(1))
        round_no = int(round_m.group(1))
        state = state_m.group(1)
        accepted = acc_m.group(1) == "true" if acc_m else None

        if round_no == 0 and state == "baseline":
            current_value = total
            continue
        if round_no == 0:
            continue

        delta_true = total - current_value
        rows.append(
            {
                "round": round_no,
                "sim_time": sim_time,
                "candidate_total": total,
                "accepted": accepted,
                "delta_true": delta_true,
            }
        )
        if accepted:
            current_value = total
    return rows


def reconstruct_history(rows, pmax, alpha, delta_eps):
    bad_rate = 0.0
    out = []
    for row in rows:
        bad_signal = 1.0 if row["delta_true"] < -delta_eps else 0.0
        bad_rate = alpha * bad_signal + (1.0 - alpha) * bad_rate
        p_value = 0.5 + (pmax - 0.5) * (1.0 - bad_rate)
        out.append(
            {
                "round": row["round"],
                "sim_time": row["sim_time"],
                "candidate_total": row["candidate_total"],
                "accepted": row["accepted"],
                "delta_true": row["delta_true"],
                "bad_signal": bad_signal,
                "bad_rate_ewma": bad_rate,
                "direction_prob_cap": p_value,
            }
        )
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct adaptive direction-cap history from tuner.log")
    parser.add_argument("--id", type=int, required=True, help="experiment id under mix/")
    parser.add_argument("--pmax", type=float, required=True, help="pmax used by the experiment")
    parser.add_argument("--alpha", type=float, default=0.15, help="EWMA alpha, default 0.15")
    parser.add_argument("--delta-eps", type=float, default=0.01, help="significance threshold for bad steps, default 0.01")
    parser.add_argument("--output-prefix", default=None, help="optional output prefix under analysis/")
    args = parser.parse_args()

    log_path = resolve_tuner_log(args.id)
    rows = parse_round_records(log_path)
    history = reconstruct_history(rows, args.pmax, args.alpha, args.delta_eps)

    prefix = args.output_prefix or ("direction_cap_history_exp_%s" % args.id)
    csv_path = SCRIPT_DIR / (prefix + ".csv")
    png_path = SCRIPT_DIR / (prefix + ".png")
    history.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    ax1.plot(history["round"], history["direction_prob_cap"], color="#1f77b4", linewidth=1.8, label="direction_prob_cap")
    ax1.set_xlabel("Tuning Round")
    ax1.set_ylabel("Direction Prob Cap", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(history["round"], history["bad_rate_ewma"], color="#d62728", linewidth=1.4, linestyle="--", label="bad_rate_ewma")
    ax2.set_ylabel("Bad Rate EWMA", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title("Adaptive direction-cap history (exp %s)" % args.id)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print("csv = %s" % csv_path)
    print("plot = %s" % png_path)
    print()
    print(history.to_string(index=False))


if __name__ == "__main__":
    main()
