#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAFFIC_DIR = ROOT_DIR / "traffic_gen"
MIX_DIR = ROOT_DIR / "mix"


class Flow:
    dport_map = defaultdict(lambda: 10000)

    def __init__(self, src, dst, size_bytes, start_time_s):
        self.src = src
        self.dst = dst
        self.size_bytes = max(int(size_bytes), 100)
        self.start_time_s = float(start_time_s)
        self.dport = Flow.dport_map[(src, dst)]
        Flow.dport_map[(src, dst)] += 1

    def __str__(self):
        return "%d %d 3 %d %d %.9f" % (
            self.src,
            self.dst,
            self.dport,
            self.size_bytes,
            self.start_time_s,
        )


def estimate_alltoall_on_time(num_workers, flow_size_bytes, bandwidth_gbps):
    bandwidth_bps = bandwidth_gbps * 1e9
    bytes_per_worker = (num_workers - 1) * flow_size_bytes
    return bytes_per_worker * 8.0 / bandwidth_bps


def spread_workers_for_topo_s(worker_count):
    # topo_s has 64 hosts, 8 hosts per ToR.
    # Spread workers as evenly as possible across 8 ToRs.
    if worker_count <= 0 or worker_count > 64:
        raise ValueError("topo_s worker_count must be between 1 and 64")

    tor_count = 8
    hosts_per_tor = 8
    base = worker_count // tor_count
    extra = worker_count % tor_count
    workers = []
    for tor_index in range(tor_count):
        take = base + (1 if tor_index < extra else 0)
        start_host = tor_index * hosts_per_tor
        workers.extend(range(start_host, start_host + take))
    if len(workers) != worker_count:
        raise ValueError("failed to allocate requested worker_count on topo_s")
    return workers


def generate_onoff_alltoall_flows(
    workers,
    base_time_s,
    num_rounds,
    flow_size_bytes,
    off_time_s,
    bandwidth_gbps,
    period_override_s=None,
):
    on_time_s = estimate_alltoall_on_time(len(workers), flow_size_bytes, bandwidth_gbps)
    period_s = period_override_s if period_override_s is not None else (on_time_s + off_time_s)

    flows = []
    round_starts = []
    for round_index in range(num_rounds):
        round_time_s = base_time_s + round_index * period_s
        round_starts.append(round_time_s)
        for src in workers:
            for dst in workers:
                if src == dst:
                    continue
                flows.append(Flow(src, dst, flow_size_bytes, round_time_s))
    flows.sort(key=lambda flow: (flow.start_time_s, flow.src, flow.dst, flow.dport))
    return flows, round_starts, on_time_s, period_s


def write_flow_file(flows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("%d\n" % len(flows))
        for flow in flows:
            handle.write("%s\n" % flow)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a topo_s LLM ON-OFF all-to-all workload")
    parser.add_argument("--workers", type=int, default=20, help="number of workers participating in all-to-all")
    parser.add_argument("--rounds", type=int, default=5, help="number of ON-OFF rounds to generate")
    parser.add_argument("--base-time", type=float, default=2.0, help="first round start time in seconds")
    parser.add_argument("--flow-size-mb", type=float, default=12.0, help="per-flow size in MiB")
    parser.add_argument("--off-ms", type=float, default=20.0, help="OFF time between rounds in ms")
    parser.add_argument("--bw-gbps", type=float, default=100.0, help="per-worker NIC bandwidth in Gbps")
    parser.add_argument("--period-ms", type=float, default=None, help="optional manual round period in ms")
    parser.add_argument(
        "--output",
        default=str(TRAFFIC_DIR / "flow_llm_onoff_topo_s_20w_5r.txt"),
        help="output flow file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.flow_size_mb <= 0 or args.off_ms < 0 or args.bw_gbps <= 0:
        raise SystemExit("flow size, off time, and bandwidth must be positive")

    workers = spread_workers_for_topo_s(args.workers)
    flow_size_bytes = int(args.flow_size_mb * 1024 * 1024)
    off_time_s = args.off_ms / 1000.0
    period_override_s = None if args.period_ms is None else args.period_ms / 1000.0

    flows, round_starts, on_time_s, period_s = generate_onoff_alltoall_flows(
        workers=workers,
        base_time_s=args.base_time,
        num_rounds=args.rounds,
        flow_size_bytes=flow_size_bytes,
        off_time_s=off_time_s,
        bandwidth_gbps=args.bw_gbps,
        period_override_s=period_override_s,
    )

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (ROOT_DIR / output_path).resolve()
    write_flow_file(flows, output_path)

    print("output = %s" % output_path)
    print("workers = %s" % " ".join(str(worker) for worker in workers))
    print("rounds = %d" % args.rounds)
    print("flows_per_round = %d" % (len(workers) * (len(workers) - 1)))
    print("total_flows = %d" % len(flows))
    print("flow_size_bytes = %d" % flow_size_bytes)
    print("estimated_on_time_s = %.9f" % on_time_s)
    print("off_time_s = %.9f" % off_time_s)
    print("period_s = %.9f" % period_s)
    for round_index, round_start in enumerate(round_starts, start=1):
        print("round_%02d_start = %.9f" % (round_index, round_start))


if __name__ == "__main__":
    main()
