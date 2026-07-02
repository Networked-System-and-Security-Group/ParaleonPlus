# Paraleon NS-3 Simulator

This repository contains an ns-3.17-based RDMA datacenter simulator for Paraleon and Paraleon+ style online parameter tuning on top of the DCQCN path. It originates from the HPCC ns-3 codebase and keeps the Paraleon runtime loop:

`scratch/tuning.py -> mix/parameter.txt -> scratch/third.cc`

## Relationship to the Original Paraleon Repository

This repository is based on the original public Paraleon simulator:

<https://github.com/czt8888/Paraleon-ns3>

On top of the original Paraleon design, we add `paraleon_plus`.

The main difference is in runtime monitoring. The original Paraleon relies on switch-side flow-size observation, while `paraleon_plus` uses host-side flow-progress
observation to infer the large-flow versus small-flow distribution. In our implementation, the simulator exports a host WR-table style report, and the tuner uses that report
instead of the original switch sketch path.

`paraleon` and `paraleon_plus` share the same tuning entry point, `scratch/tuning.py`, but they use different observation inputs. `paraleon` reads switch-side sketch
reports, while `paraleon_plus` reads host-side flow reports. The tuning core is shared, while the monitoring path is different.

Compared with the conference-version Paraleon, `paraleon_plus` also reflects the journal-version refinements to the tuning workflow, including the host-side monitoring
design and the improved search behavior for adaptive parameter tuning.

## Quick Start

### 1. Build

```bash
./waf configure
./waf build
```

If your default compiler is too new for this ns-3.17 codebase, retry configure with:

```bash
CC='gcc-5' CXX='g++-5' ./waf configure
```

### 2. Run an experiment

```bash
python run.py --scheme paraleon --topo topo_s.txt --flow-file test_flow_topo_s.txt --cc dcqcn --sim-time 2.1 --msg run
```

The most important arguments are:

- `--scheme`: selects the control path. Common choices are `paraleon` and `paraleon_plus`.
- `--flow-file`: selects the workload input file. Relative paths are resolved under `mix/`.
- `--topo`: selects the topology input file. Relative paths are resolved under `mix/`.

Example:

```bash
python run.py --scheme paraleon_plus --topo topo_s.txt --flow-file test_flow_topo_s.txt --cc dcqcn --sim-time 2.1 --msg plus-run
```

### 3. Flow file format

Use [mix/test_flow_topo_s.txt](mix/test_flow_topo_s.txt) as the reference example.

The parser expects:

- First line: total number of flows.
- Each following line: `src dst pg dport size_bytes start_time_seconds`

In the simulator implementation, the fifth field is interpreted as the flow size in bytes.

### 4. Topology file format

Use [mix/topo_s.txt](mix/topo_s.txt) as the reference example.

The parser expects:

- First line: `node_count switch_count link_count`
- Second line: a list of switch node IDs
- Remaining lines: `src dst data_rate link_delay error_rate`

Each remaining line describes one directed link in the topology.

### 5. Experiment artifacts

Each `run.py` invocation creates one run directory under `mix/`, named as:

`mix/<id>-<timestamp>-<msg>/`

The most important files in each run directory are:

- `manifest.json`: the canonical summary of the run, including selected inputs, generated outputs, and return codes
- `config.txt`: the generated ns-3 configuration passed to `scratch/third.cc`
- selected topology and flow input paths recorded in `manifest.json`
- `parameter.txt`: the live parameter file written by the tuner and reloaded by the simulator
- `simulator.log`: stdout/stderr from the ns-3 simulator
- `tuner.log`: stdout/stderr from `scratch/tuning.py`
- `fct_*.txt`, `pfc.txt`, `qlen_*.txt`, `trace_*.tr`: core output traces
- `monitor_output.tr` and `metric_output.tr`: tuner-side monitoring and utility traces

## Repository Layout

- `run.py`: experiment wrapper that creates a run directory and launches the simulator and tuner
- `scratch/third.cc`: main ns-3 simulation entry point
- `scratch/tuning.py`: online tuning loop
- `mix/`: sample inputs and per-run outputs

## Upstream Base

This simulator is built on top of the HPCC ns-3 codebase:

<https://github.com/alibaba-edu/High-Precision-Congestion-Control>
