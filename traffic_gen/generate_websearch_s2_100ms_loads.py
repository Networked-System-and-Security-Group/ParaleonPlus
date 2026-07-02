#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import gen_traffic


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOADS = [50, 60, 70]
DEFAULT_DURATION_S = 0.1
DEFAULT_BASE_TIME_S = 2.0
DEFAULT_HOSTS = list(range(64))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WebSearch s2 workloads for multiple host-rate loads")
    parser.add_argument("--loads", default="50,60,70", help="comma-separated per-host loads in Gbps")
    parser.add_argument("--duration-ms", type=float, default=100.0, help="traffic duration in milliseconds")
    parser.add_argument("--base-time-s", type=float, default=DEFAULT_BASE_TIME_S, help="base start time in seconds")
    parser.add_argument("--seed-base", type=int, default=35000, help="base random seed; each load uses seed_base + load")
    return parser.parse_args()


def parse_loads(raw_value):
    loads = []
    seen = set()
    for part in str(raw_value).split(","):
        token = part.strip()
        if not token:
            continue
        load = int(token)
        if load in seen:
            continue
        seen.add(load)
        loads.append(load)
    if not loads:
        raise SystemExit("no loads were provided")
    return loads


def generate_one(load_gbps, duration_s, base_time_s, seed_base):
    cdf_path = SCRIPT_DIR / "WebSearch.txt"
    output_path = SCRIPT_DIR / ("flow_websearch_s2_100ms_%dG.txt" % load_gbps)

    random.seed(seed_base + load_gbps)
    gen_traffic.dport_map.clear()
    flows = gen_traffic.generate_flows(
        DEFAULT_HOSTS,
        DEFAULT_HOSTS,
        str(cdf_path),
        "%dG" % load_gbps,
        duration_s,
        base_time=base_time_s,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("%d\n" % len(flows))
        for flow in flows:
            handle.write(str(flow) + "\n")

    total_bytes = sum(flow.size for flow in flows)
    aggregate_rate_gbps = total_bytes * 8 / duration_s / 1e9 if duration_s > 0 else 0
    print(
        "generated load=%dG flows=%d avg_size=%.2fB agg_rate=%.3fGbps output=%s"
        % (load_gbps, len(flows), total_bytes / len(flows), aggregate_rate_gbps, output_path)
    )


def main():
    args = parse_args()
    loads = parse_loads(args.loads)
    duration_s = args.duration_ms / 1000.0
    for load_gbps in loads:
        generate_one(load_gbps, duration_s, args.base_time_s, args.seed_base)


if __name__ == "__main__":
    main()
