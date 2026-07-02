from collections import defaultdict

# 每个 src-dst pair 的 dport 从 10000 开始递增
dport_map = defaultdict(lambda: 10000)


class Flow:
    def __init__(self, src, dst, size, t):
        self.src = src
        self.dst = dst
        self.size = max(size, 100)
        self.t = t
        self.dport = dport_map[(src, dst)]
        dport_map[(src, dst)] += 1

    def __str__(self):
        return f"{self.src} {self.dst} 3 {self.dport} {self.size} {self.t:.9f}"


def estimate_alltoall_on_time(
    num_workers,
    flow_size_bytes,
    bandwidth_gbps=100
):
    """
    估计一轮 all-to-all 的 ON 阶段理论完成时间。

    假设：
    1. 每个节点 NIC 带宽为 bandwidth_gbps
    2. 全双工
    3. 瓶颈主要在单节点发送/接收带宽
    4. 不考虑交换机拥塞、incast 排队、协议开销
    """
    bandwidth_bps = bandwidth_gbps * 1e9

    # 每个节点需要向其他 N-1 个节点发送 flow_size_bytes
    bytes_per_node = (num_workers - 1) * flow_size_bytes

    on_time = bytes_per_node * 8 / bandwidth_bps
    return on_time


def generate_onoff_alltoall_flows(
    workers,
    base_t,
    num_rounds,
    flow_size_bytes=12 * 1024 * 1024,
    off_time=0.020,
    bandwidth_gbps=100,
    period_override=None
):
    """
    生成多轮 ON/OFF all-to-all workload。

    参数：
    workers: worker 节点列表，例如 [0, 1, ..., 19]
    base_t: 第一轮开始时间
    num_rounds: 总轮数
    flow_size_bytes: 每条 flow 大小，默认 12MiB
    off_time: OFF 阶段模型更新时间，默认 20ms
    bandwidth_gbps: 单节点带宽，默认 100Gbps
    period_override: 如果你想手动指定每轮间隔，可以传入该值，单位秒

    返回：
    flows: 所有 Flow 对象
    round_starts: 每轮开始时间
    period: 每轮间隔
    """

    assert len(workers) == 20, "This workload expects exactly 20 workers."

    if period_override is None:
        on_time = estimate_alltoall_on_time(
            num_workers=len(workers),
            flow_size_bytes=flow_size_bytes,
            bandwidth_gbps=bandwidth_gbps
        )
        period = on_time + off_time
    else:
        period = period_override

    flows = []
    round_starts = []

    for r in range(num_rounds):
        round_t = base_t + r * period
        round_starts.append(round_t)

        # ON 阶段：所有 worker 同时 all-to-all
        for src in workers:
            for dst in workers:
                if src == dst:
                    continue

                flows.append(
                    Flow(
                        src=src,
                        dst=dst,
                        size=flow_size_bytes,
                        t=round_t
                    )
                )

    return flows, round_starts, period


def write_flows_to_file(flows, filename):
    with open(filename, "w") as f:
        for flow in flows:
            f.write(str(flow) + "\n")

workers = [
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19
]

flows, round_starts, period = generate_onoff_alltoall_flows(
    workers=workers,
    base_t=0.0,
    num_rounds=10,
    flow_size_bytes=12 * 1024 * 1024,
    off_time=0.020,
    bandwidth_gbps=100
)

write_flows_to_file(flows, "/home/zhangj25/dcqcn-tuning/Paraleon-ns3/traffic_gen/llm_traffic.txt")

print(f"Total flows: {len(flows)}")
print(f"Flows per round: {20 * 19}")
print(f"Round period: {period:.9f} s")
print("Round starts:")
for i, t in enumerate(round_starts):
    print(i, f"{t:.9f}")
