import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MIX_DIR = ROOT_DIR / "mix"
DEFAULT_ACC_MODEL_PATH = ROOT_DIR / "models" / "acc_default_base_model.npz"
DEFAULT_TOPOLOGY_FILE = "topo_s.txt"
DEFAULT_FLOW_FILE = "flow_normal_s2.txt"
DEFAULT_TUNER_WEIGHTS_AGG = (0.6, 0.2, 0.2)
DEFAULT_TUNER_WEIGHTS_CON = (0.3, 0.5, 0.2)
DEFAULT_DIRECTION_PROB_CAP = 0.7
DEFAULT_TUNER_INITIAL_TEMPERATURE = 0.1
DEFAULT_TUNER_FINAL_TEMPERATURE = 0.01
DEFAULT_TUNER_COOLING_RATE = 0.76
DEFAULT_TUNER_ATTEMPT_TIMES = 8
DEFAULT_TUNER_MUTATE_COUNT = 8
DEFAULT_MONITOR_INTERVAL_MS = 1.0
DEFAULT_PARAMETER_TUNING_INTERVAL_MS = 1.002
DEFAULT_PARALEON_PLUS_CHUNK_KB = 1024
EXPERT_RATE_AI_MBPS = 50
EXPERT_RATE_HAI_MBPS = 150
EXPERT_RATE_DECREASE_INTERVAL_US = 80
EXPERT_MIN_TIME_BETWEEN_CNPS_US = 96
EXPERT_KMIN_KB = 1600
EXPERT_KMAX_KB = 6400
EXPERT_PMAX = 0.2


CONFIG_TEMPLATE = """ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE {topology_file}
FLOW_FILE {flow_file}
TRACE_FILE {trace_file}
TRACE_OUTPUT_FILE {trace_output_file}
FCT_OUTPUT_FILE {fct_output_file}
PFC_OUTPUT_FILE {pfc_output_file}

SIMULATOR_STOP_TIME {simulator_stop_time}

CC_MODE {mode}
ALPHA_RESUME_INTERVAL {t_alpha}
RATE_DECREASE_INTERVAL {t_dec}
CLAMP_TARGET_RATE 0
RP_TIMER {t_inc}
EWMA_GAIN {g}
FAST_RECOVERY_TIMES 1
RATE_AI {ai}Mb/s
RATE_HAI {hai}Mb/s
RATE_ON_FIRST_CNP {rate_on_first_cnp}
MIN_RATE {min_rate}Mb/s
DCTCP_RATE_AI {dctcp_ai}Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN {has_win}
GLOBAL_T 1
VAR_WIN {vwin}
FAST_REACT {us}
U_TARGET {u_tgt}
MI_THRESH {mi}
INT_MULTI {int_multi}
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE {pint_log_base}
PINT_PROB {pint_prob}

RATE_BOUND 1

ACK_HIGH_PRIO {ack_prio}

LINK_DOWN {link_down}

ENABLE_TRACE {enable_tr}

KMAX_MAP {kmax_map}
KMIN_MAP {kmin_map}
PMAX_MAP {pmax_map}
BUFFER_SIZE {buffer_size_mb}
QLEN_MON_FILE {qlen_mon_file}
QLEN_MON_START {qlen_mon_start}
QLEN_MON_END {qlen_mon_end}
QLEN_MON_INTERVAL {qlen_mon_interval}
SKETCH_MON_INTERVAL {sketch_mon_interval}
ACC_MON_INTERVAL {acc_mon_interval}
PARAMETER_TUNING_START {parameter_tuning_start}
PARAMETER_TUNING_END {parameter_tuning_end}
PARAMETER_TUNING_INTERVAL {parameter_tuning_interval}
"""

def resolve_optional_path(path_value):
	if not path_value:
		return None
	return str(Path(path_value).expanduser().resolve())


def resolve_acc_model_path(args):
	if args.scheme != "acc":
		return None
	if args.acc_model:
		return resolve_optional_path(args.acc_model)
	if args.acc_from_scratch:
		return None
	if DEFAULT_ACC_MODEL_PATH.exists():
		return str(DEFAULT_ACC_MODEL_PATH.resolve())
	return None


def parse_weight_triplet(raw_value):
	parts = [part.strip() for part in str(raw_value).split(",")]
	if len(parts) != 3:
		raise argparse.ArgumentTypeError("weight triplet must have exactly three comma-separated values")

	weights = []
	for part in parts:
		try:
			weight = float(part)
		except ValueError as exc:
			raise argparse.ArgumentTypeError("invalid weight value: %s" % part) from exc
		if weight < 0 or weight > 1:
			raise argparse.ArgumentTypeError("weight must be between 0 and 1: %s" % part)
		weights.append(weight)

	if abs(sum(weights) - 1.0) > 1e-6:
		raise argparse.ArgumentTypeError("weights must sum to 1.0")
	return tuple(weights)


def parse_unit_interval(raw_value):
	try:
		value = float(raw_value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("value must be a float between 0 and 1") from exc
	if value < 0 or value > 1:
		raise argparse.ArgumentTypeError("value must be between 0 and 1")
	return value


def parse_open_unit_interval(raw_value):
	try:
		value = float(raw_value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("value must be a float between 0 and 1") from exc
	if value <= 0 or value >= 1:
		raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
	return value


def parse_positive_float(raw_value):
	try:
		value = float(raw_value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("value must be a positive float") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError("value must be positive")
	return value


def parse_nonnegative_float(raw_value):
	try:
		value = float(raw_value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("value must be a nonnegative float") from exc
	if value < 0:
		raise argparse.ArgumentTypeError("value must be nonnegative")
	return value


def parse_positive_int(raw_value):
	try:
		value = int(raw_value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("value must be a positive integer") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError("value must be positive")
	return value


def format_weight_triplet(weights):
	return ",".join("%.6g" % weight for weight in weights)


def slugify_message(message):
	cleaned = re.sub(r"[^\w-]+", "-", message.strip(), flags=re.UNICODE)
	cleaned = re.sub(r"-+", "-", cleaned).strip("-")
	return cleaned or "run"


def next_run_id():
	run_ids = []
	for path in MIX_DIR.iterdir():
		if not path.is_dir():
			continue
		prefix = path.name.split("-", 1)[0]
		if prefix.isdigit():
			run_ids.append(int(prefix))
	return max(run_ids, default=0) + 1


def create_run_dir(message):
	run_id = next_run_id()
	timestamp = datetime.now().strftime("%d:%H:%M")
	run_dir = MIX_DIR / f"{run_id}-{timestamp}-{slugify_message(message)}"
	run_dir.mkdir(parents=True, exist_ok=False)
	return run_id, run_dir


def write_text(path, content):
	path.write_text(content, encoding="utf-8")


def write_json(path, payload):
	path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_config_path(path):
	return path.resolve().as_posix()


def resolve_input_path(path_value, default_name):
	if path_value is None:
		candidate = Path(default_name)
	else:
		candidate = Path(path_value).expanduser()

	if candidate.is_absolute():
		resolved = candidate.resolve()
	elif candidate.parts and candidate.parts[0] in (".", ".."): 
		resolved = (ROOT_DIR / candidate).resolve()
	else:
		resolved = (MIX_DIR / candidate).resolve()

	if not resolved.exists():
		raise SystemExit("input file not found: %s" % resolved)
	return resolved


def inspect_topology_file(topology_path):
	with topology_path.open("r", encoding="utf-8") as topology_file:
		header = topology_file.readline().split()
		if len(header) != 3:
			raise SystemExit("invalid topology header in %s" % topology_path)
		node_count, switch_count, link_count = (int(value) for value in header)
		switch_ids = topology_file.readline().split()
		if len(switch_ids) != switch_count:
			raise SystemExit("switch count mismatch in %s" % topology_path)
	return {
		"name": topology_path.name,
		"path": str(topology_path),
		"node_count": node_count,
		"switch_count": switch_count,
		"link_count": link_count,
		"switch_ids": [int(node_id) for node_id in switch_ids],
	}


def inspect_flow_file(flow_path):
	with flow_path.open("r", encoding="utf-8") as flow_file:
		header = flow_file.readline().strip()
		if not header:
			raise SystemExit("empty flow file: %s" % flow_path)
		try:
			flow_count = int(header)
		except ValueError as exc:
			raise SystemExit("invalid flow header in %s" % flow_path) from exc
	return {
		"name": flow_path.name,
		"path": str(flow_path),
		"flow_count": flow_count,
	}


def build_trace_file(trace_path, topology_metadata):
	node_ids = list(range(topology_metadata["node_count"]))
	lines = [str(len(node_ids)), " ".join(str(node_id) for node_id in node_ids)]
	write_text(trace_path, "\n".join(lines) + "\n")


def build_cc_profile(args):
	bw = args.bw
	kmax_map = "2 %d %d %d %d" % (bw * 1_000_000_000, 400 * bw / 25, bw * 4 * 1_000_000_000, 400 * bw * 4 / 25)
	kmin_map = "2 %d %d %d %d" % (bw * 1_000_000_000, 100 * bw / 25, bw * 4 * 1_000_000_000, 100 * bw * 4 / 25)
	pmax_map = "2 %d %.2f %d %.2f" % (bw * 1_000_000_000, 0.2, bw * 4 * 1_000_000_000, 0.2)

	profile = {
		"kmax_map": kmax_map,
		"kmin_map": kmin_map,
		"pmax_map": pmax_map,
		"pint_log_base": args.pint_log_base,
		"pint_prob": args.pint_prob,
		"mi": args.mi,
		"u_tgt": args.utgt / 100.0,
		"link_down": args.down,
		"enable_tr": args.enable_tr,
		"buffer_size_mb": args.buffer_mb,
	}

	# if args.scheme == "dcqcn_plus":
	# 	profile["kmax_map"] = "2 %d %d %d %d" % (bw * 1_000_000_000, 200, bw * 4 * 1_000_000_000, 200)
	# 	profile["kmin_map"] = "2 %d %d %d %d" % (bw * 1_000_000_000, 20, bw * 4 * 1_000_000_000, 20)

	if args.cc == "dcqcn":
		profile.update({"cc_label": args.cc, "mode": 1, "t_alpha": 1, "t_dec": 4, "t_inc": 300, "g": 0.00390625, "ai": 5, "hai": 50, "rate_on_first_cnp": 0.5, "min_rate": 1, "dctcp_ai": 1000, "has_win": 0, "vwin": 0, "us": 0, "int_multi": 1, "ack_prio": 1})
		if args.scheme == "expert":
			profile.update(
				{
					"t_dec": EXPERT_RATE_DECREASE_INTERVAL_US,
					"ai": EXPERT_RATE_AI_MBPS,
					"hai": EXPERT_RATE_HAI_MBPS,
					"kmin_map": "2 %d %d %d %d" % (
						bw * 1_000_000_000,
						EXPERT_KMIN_KB,
						bw * 4 * 1_000_000_000,
						EXPERT_KMIN_KB,
					),
					"kmax_map": "2 %d %d %d %d" % (
						bw * 1_000_000_000,
						EXPERT_KMAX_KB,
						bw * 4 * 1_000_000_000,
						EXPERT_KMAX_KB,
					),
					"pmax_map": "2 %d %.2f %d %.2f" % (
						bw * 1_000_000_000,
						EXPERT_PMAX,
						bw * 4 * 1_000_000_000,
						EXPERT_PMAX,
					),
				}
			)
		return profile

	raise ValueError("unknown cc: %s" % args.cc)


def build_config(run_dir, profile, topology_path, flow_path, trace_path, args):
	cc_label = profile["cc_label"]
	flow_label = Path(flow_path).stem
	failure = "_down" if args.down != "0 0 0" else ""
	parameter_tuning_start = int((args.start_time + args.tuning_start_offset_ms / 1000.0) * 1_000_000_000)
	parameter_tuning_end = int(min(args.sim_time - 0.05, args.start_time + 0.5) * 1_000_000_000)
	qlen_mon_start = int((args.start_time + args.qlen_mon_start_offset_ms / 1000.0) * 1_000_000_000)
	qlen_mon_end = int(min(args.sim_time - 0.01, max(args.start_time + args.window_ms / 1000.0 + 0.05, args.start_time + 0.55)) * 1_000_000_000)
	monitor_interval = int(args.monitor_interval_ms * 1_000_000)
	parameter_tuning_interval = int(args.parameter_tuning_interval_ms * 1_000_000)
	trace_output_path = run_dir / ("trace_%s_%s%s.tr" % (flow_label, cc_label, failure))
	fct_output_path = run_dir / ("fct_%s_%s%s.txt" % (flow_label, cc_label, failure))
	if args.scheme in ["paraleon", "paraleon_plus"]:
		pfc_output_path = run_dir / "pfc.txt"
	else:
		pfc_output_path = run_dir / ("pfc_%s_%s%s.txt" % (flow_label, cc_label, failure))
	qlen_output_path = run_dir / ("qlen_%s_%s%s.txt" % (flow_label, cc_label, failure))

	config_path = run_dir / "config.txt"
	config_text = CONFIG_TEMPLATE.format(
		topology_file=to_config_path(topology_path),
		flow_file=to_config_path(flow_path),
		trace_file=to_config_path(trace_path),
		trace_output_file=to_config_path(trace_output_path),
		fct_output_file=to_config_path(fct_output_path),
		pfc_output_file=to_config_path(pfc_output_path),
		simulator_stop_time="%.2f" % args.sim_time,
		mode=profile["mode"],
		t_alpha=profile["t_alpha"],
		t_dec=profile["t_dec"],
		t_inc=profile["t_inc"],
		g=profile["g"],
		ai=profile["ai"],
		hai=profile["hai"],
		rate_on_first_cnp=profile["rate_on_first_cnp"],
		min_rate=profile["min_rate"],
		dctcp_ai=profile["dctcp_ai"],
		has_win=profile["has_win"],
		vwin=profile["vwin"],
		us=profile["us"],
		u_tgt=profile["u_tgt"],
		mi=profile["mi"],
		int_multi=profile["int_multi"],
		pint_log_base=profile["pint_log_base"],
		pint_prob=profile["pint_prob"],
		ack_prio=profile["ack_prio"],
		link_down=profile["link_down"],
		enable_tr=profile["enable_tr"],
		kmax_map=profile["kmax_map"],
		kmin_map=profile["kmin_map"],
		pmax_map=profile["pmax_map"],
		buffer_size_mb=profile["buffer_size_mb"],
		qlen_mon_file=to_config_path(qlen_output_path),
		qlen_mon_start=qlen_mon_start,
		qlen_mon_end=qlen_mon_end,
		qlen_mon_interval=monitor_interval,
		sketch_mon_interval=monitor_interval,
		acc_mon_interval=monitor_interval,
		parameter_tuning_start=parameter_tuning_start,
		parameter_tuning_end=parameter_tuning_end,
		parameter_tuning_interval=parameter_tuning_interval,
	)
	write_text(config_path, config_text)
	return config_path, {
		"cc_label": cc_label,
		"failure_suffix": failure,
		"parameter_tuning_start": parameter_tuning_start,
		"parameter_tuning_end": parameter_tuning_end,
		"qlen_mon_start": qlen_mon_start,
		"qlen_mon_end": qlen_mon_end,
		"trace_output": str(trace_output_path),
		"fct_output": str(fct_output_path),
		"pfc_output": str(pfc_output_path),
		"qlen_output": str(qlen_output_path),
	}


def move_runtime_artifacts(run_dir):
	for artifact_name in ["nic_datarate.tr", "packet_path.tr", "nic_cnp_received.tr"]:
		source_path = MIX_DIR / artifact_name
		if source_path.exists():
			shutil.move(str(source_path), str(run_dir / artifact_name))


def launch_process(command, log_path, env):
	log_file = open(log_path, "w")
	process = subprocess.Popen(command, cwd=str(ROOT_DIR), env=env, stdout=log_file, stderr=subprocess.STDOUT)
	return process, log_file


def resolve_simulator_command(config_path):
	built_binary = ROOT_DIR / "build" / "scratch" / "third"
	if built_binary.exists() and built_binary.is_file():
		return [str(built_binary), config_path.resolve().as_posix()]
	return [str(ROOT_DIR / "waf"), "--run", "scratch/third %s" % config_path.resolve().as_posix()]


def extend_simulator_library_path(env):
	build_dir = str((ROOT_DIR / "build").resolve())
	existing = env.get("LD_LIBRARY_PATH")
	if existing:
		env["LD_LIBRARY_PATH"] = build_dir + os.pathsep + existing
	else:
		env["LD_LIBRARY_PATH"] = build_dir


def terminate_process(process):
	if process is None or process.poll() is not None:
		return None
	process.terminate()
	try:
		process.wait(timeout=5)
	except subprocess.TimeoutExpired:
		process.kill()
		process.wait(timeout=5)
	return process.returncode


def build_controller_command(args, python_executable, run_dir):
	if args.no_tuner:
		return None, None

	if args.scheme in ["paraleon", "paraleon_plus"]:
		return [python_executable, str(ROOT_DIR / "scratch" / "tuning.py")], run_dir / "tuner.log"

	if args.scheme == "acc":
		acc_model_path = resolve_acc_model_path(args)
		command = [
			python_executable,
			str(ROOT_DIR / "scratch" / "acc_controller.py"),
			"--run-dir",
			str(run_dir.resolve()),
		]
		if acc_model_path:
			command.extend(["--model", acc_model_path])
		if args.acc_freeze_model:
			command.append("--freeze-model")
		if args.acc_init_params:
			command.extend(["--init-params", resolve_optional_path(args.acc_init_params)])
		return command, run_dir / "controller.log"

	if args.scheme in ["dcqcn_plus", "expert", "fixed_params"]:
		return None, None

	raise ValueError("unsupported scheme: %s" % args.scheme)


def ensure_parameter_file(run_dir, controller_command):
	if controller_command is not None:
		return
	parameter_path = run_dir / "parameter.txt"
	if not parameter_path.exists():
		write_text(parameter_path, "")


def apply_scheme_specific_parameter_defaults(args, run_dir):
	if args.scheme == "fixed_params":
		if not args.fixed_params_file:
			raise SystemExit("--fixed-params-file is required when --scheme fixed_params")
		source_path = Path(resolve_optional_path(args.fixed_params_file))
		if not source_path.exists():
			raise SystemExit("fixed params file not found: %s" % source_path)
		parameter_path = run_dir / "parameter.txt"
		write_text(parameter_path, source_path.read_text(encoding="utf-8"))
		return

	if args.scheme != "expert":
		return
	parameter_path = run_dir / "parameter.txt"
	parameter_lines = [
		"ai_rate=%d" % EXPERT_RATE_AI_MBPS,
		"hai_rate=%d" % EXPERT_RATE_HAI_MBPS,
		"kmin=%d" % EXPERT_KMIN_KB,
		"kmax=%d" % EXPERT_KMAX_KB,
		"pmax=%.6g" % EXPERT_PMAX,
	]
	write_text(parameter_path, "\n".join(parameter_lines) + "\n")


def parse_args():
	parser = argparse.ArgumentParser(description="Run Paraleon experiments with topology and flow inputs from mix/")
	parser.add_argument("--scheme", default="paraleon", choices=["paraleon", "paraleon_plus", "acc", "dcqcn_plus", "expert", "fixed_params"], help="control scheme to run on top of the shared experiment shell")
	parser.add_argument("--cc", default="dcqcn", help="hp/dcqcn/dcqcn_paper/dcqcn_vwin/dcqcn_paper_vwin/timely/timely_vwin/dctcp/hpccPint")
	parser.add_argument("--topo", default=DEFAULT_TOPOLOGY_FILE, help="topology input file name or path; relative paths are resolved under mix/")
	parser.add_argument("--flow-file", default=DEFAULT_FLOW_FILE, help="flow input file name or path; relative paths are resolved under mix/")
	parser.add_argument("--fixed-params-file", default=None, help="parameter file used when --scheme fixed_params")
	parser.add_argument("--bw", type=int, default=100, help="link bandwidth in Gbps")
	parser.add_argument("--buffer-mb", type=int, default=12, help="switch buffer size in MB")
	parser.add_argument("--window-ms", type=int, default=100, help="active workload generation window in milliseconds")
	parser.add_argument("--start-time", type=float, default=2.0, help="when generated flows begin, in seconds")
	parser.add_argument("--sim-time", type=float, default=2.6, help="simulation stop time in seconds")
	parser.add_argument("--monitor-interval-ms", type=parse_positive_float, default=DEFAULT_MONITOR_INTERVAL_MS, help="monitoring interval in milliseconds for throughput/RTT/queue snapshots")
	parser.add_argument("--parameter-tuning-interval-ms", type=parse_positive_float, default=DEFAULT_PARAMETER_TUNING_INTERVAL_MS, help="parameter update interval in milliseconds")
	parser.add_argument("--qlen-mon-start-offset-ms", type=parse_nonnegative_float, default=13.0, help="offset in ms from start-time for queue/throughput/RTT monitoring to begin")
	parser.add_argument("--tuning-start-offset-ms", type=parse_nonnegative_float, default=15.0, help="offset in ms from start-time for parameter reloading to begin")
	parser.add_argument("--msg", default="run", help="extra text appended to the run directory name")
	parser.add_argument("--down", default="0 0 0", help="link down event")
	parser.add_argument("--utgt", type=int, default=95, help="eta of HPCC")
	parser.add_argument("--mi", type=int, default=0, help="MI_THRESH")
	parser.add_argument("--hpai", type=int, default=0, help="AI for HPCC")
	parser.add_argument("--pint_log_base", type=float, default=1.01, help="PINT log base")
	parser.add_argument("--pint_prob", type=float, default=1.0, help="PINT sampling probability")
	parser.add_argument("--enable_tr", type=int, default=0, help="enable packet-level event dump")
	parser.add_argument("--skip-build", action="store_true", help="skip running waf before launching the experiment; requires an already-built simulator")
	parser.add_argument("--no-tuner", action="store_true", help="skip tuning.py and only run the simulator")
	parser.add_argument("--collect-only", action="store_true", help="run tuning.py in observe-only mode: collect metrics and utility without changing parameters")
	parser.add_argument("--aggressive-weights", type=parse_weight_triplet, default=DEFAULT_TUNER_WEIGHTS_AGG, help="three comma-separated aggressive utility weights: throughput,rtt,pfc")
	parser.add_argument("--conservative-weights", type=parse_weight_triplet, default=DEFAULT_TUNER_WEIGHTS_CON, help="three comma-separated conservative utility weights: throughput,rtt,pfc")
	parser.add_argument("--direction-prob-cap", type=parse_unit_interval, default=DEFAULT_DIRECTION_PROB_CAP, help="cap used in tuning.py when converting large/small flow ratios to direction probabilities")
	parser.add_argument("--direction-cap-policy", choices=["fixed", "adaptive_ewma"], default="adaptive_ewma", help="direction probability cap policy used by tuning.py")
	parser.add_argument("--aggressive-mode-large-flow-threshold", type=parse_unit_interval, default=None, help="large-flow ratio threshold for switching to aggressive mode")
	parser.add_argument("--tuner-initial-temperature", type=parse_positive_float, default=DEFAULT_TUNER_INITIAL_TEMPERATURE, help="override the tuner initial temperature")
	parser.add_argument("--tuner-final-temperature", type=parse_positive_float, default=DEFAULT_TUNER_FINAL_TEMPERATURE, help="override the tuner final temperature")
	parser.add_argument("--tuner-cooling-rate", type=parse_open_unit_interval, default=DEFAULT_TUNER_COOLING_RATE, help="override the tuner cooling rate")
	parser.add_argument("--tuner-attempt-times", type=parse_positive_int, default=DEFAULT_TUNER_ATTEMPT_TIMES, help="override the number of candidates tried per temperature stage")
	parser.add_argument("--tuner-mutate-count", type=parse_positive_int, default=DEFAULT_TUNER_MUTATE_COUNT, help="override how many parameters mutate in each candidate")
	parser.add_argument("--tuner-step-scale", type=parse_positive_float, default=None, help="override the per-parameter mutation step scale")
	parser.add_argument("--tuner-seed", type=int, default=None, help="optional random seed for tuning.py")
	parser.add_argument("--paraleon-plus-chunk-kb", type=parse_positive_int, default=DEFAULT_PARALEON_PLUS_CHUNK_KB, help="chunk size in KiB used by Paraleon+ WR-table observation")
	parser.add_argument("--acc-model", default=None, help="optional offline pretrained ACC model bundle (.npz); when omitted ACC uses the default base model if available")
	parser.add_argument("--acc-from-scratch", action="store_true", help="ignore the default ACC base model and start ACC training from scratch unless --acc-model is provided")
	parser.add_argument("--acc-freeze-model", action="store_true", help="when loading --acc-model, keep the controller in inference-only mode")
	parser.add_argument("--acc-init-params", default=None, help="optional ACC initial ECN parameter JSON file")
	return parser.parse_args()


def resolve_python_executable():
	venv_python = ROOT_DIR / "venv" / "bin" / "python"
	if venv_python.exists():
		return str(venv_python)
	return sys.executable


def main():
	args = parse_args()
	if args.tuner_initial_temperature is not None and args.tuner_final_temperature is not None:
		if args.tuner_final_temperature > args.tuner_initial_temperature:
			raise SystemExit("--tuner-final-temperature cannot exceed --tuner-initial-temperature")

	built_binary = ROOT_DIR / "build" / "scratch" / "third"
	if args.skip_build:
		if not built_binary.exists():
			raise SystemExit("--skip-build requires an existing built simulator at %s" % built_binary)
	else:
		if os.system("./waf") != 0:
			raise SystemExit("failed to build simulator with waf")
	if args.sim_time <= args.start_time + 0.05:
		raise SystemExit("sim-time must be at least 50ms later than start-time")
	if args.collect_only and args.no_tuner:
		raise SystemExit("--collect-only requires tuning.py, so it cannot be combined with --no-tuner")
	if args.collect_only and args.scheme not in ["paraleon", "paraleon_plus"]:
		raise SystemExit("--collect-only is only supported for paraleon and paraleon_plus")

	if args.scheme in ["acc", "dcqcn_plus", "paraleon_plus", "expert", "fixed_params"] and args.cc != "dcqcn":
		raise SystemExit("ACC, DCQCN+, and Paraleon+ currently require --cc dcqcn because they are bound to the original DCQCN path")

	run_id, run_dir = create_run_dir(args.msg)
	topology_path = resolve_input_path(args.topo, DEFAULT_TOPOLOGY_FILE)
	topology_metadata = inspect_topology_file(topology_path)
	flow_path = resolve_input_path(args.flow_file, DEFAULT_FLOW_FILE)
	flow_metadata = inspect_flow_file(flow_path)
	trace_path = run_dir / "trace.txt"

	build_trace_file(trace_path, topology_metadata)

	profile = build_cc_profile(args)
	config_path, config_metadata = build_config(run_dir, profile, topology_path, flow_path, trace_path, args)
	python_executable = resolve_python_executable()
	acc_model_path = resolve_acc_model_path(args)
	controller_command, controller_log_path = build_controller_command(args, python_executable, run_dir)
	ensure_parameter_file(run_dir, controller_command)
	apply_scheme_specific_parameter_defaults(args, run_dir)
	tuner_enabled = args.scheme in ["paraleon", "paraleon_plus"] and controller_command is not None
	controller_enabled = args.scheme == "acc" and controller_command is not None

	manifest_path = run_dir / "manifest.json"
	manifest = {
		"id": run_id,
		"run_dir": str(run_dir),
		"created_at": datetime.now().isoformat(),
		"message": args.msg,
		"scheme": args.scheme,
		"topology": dict(topology_metadata, **{"buffer_mb": args.buffer_mb}),
		"workload": flow_metadata,
		"config": dict(config_metadata, **{"path": str(config_path), "cc": args.cc, "sim_time_s": args.sim_time}),
		"files": {
			"topology": str(topology_path),
			"flow": str(flow_path),
			"trace": str(trace_path),
			"config": str(config_path),
			"parameter": str(run_dir / "parameter.txt"),
			"trace_output": config_metadata["trace_output"],
			"fct_output": config_metadata["fct_output"],
			"pfc_output": config_metadata["pfc_output"],
			"qlen_output": config_metadata["qlen_output"],
			"simulator_log": str(run_dir / "simulator.log"),
		},
		"tuner_enabled": tuner_enabled,
		"controller_enabled": controller_enabled,
	}
	if args.collect_only:
		manifest["collect_only"] = True
	if args.aggressive_weights is not None or args.conservative_weights is not None:
		manifest["tuner_weights"] = {}
		if args.aggressive_weights is not None:
			manifest["tuner_weights"]["aggressive"] = list(args.aggressive_weights)
		if args.conservative_weights is not None:
			manifest["tuner_weights"]["conservative"] = list(args.conservative_weights)
	if tuner_enabled:
		manifest["tuner_direction_prob_cap"] = args.direction_prob_cap
		manifest["tuner_config"] = {"direction_prob_cap": args.direction_prob_cap}
		manifest["tuner_config"]["direction_cap_policy"] = args.direction_cap_policy
		if args.aggressive_mode_large_flow_threshold is not None:
			manifest["tuner_config"]["aggressive_mode_large_flow_threshold"] = args.aggressive_mode_large_flow_threshold
		if args.tuner_initial_temperature is not None:
			manifest["tuner_config"]["initial_temperature"] = args.tuner_initial_temperature
		if args.tuner_final_temperature is not None:
			manifest["tuner_config"]["final_temperature"] = args.tuner_final_temperature
		if args.tuner_cooling_rate is not None:
			manifest["tuner_config"]["cooling_rate"] = args.tuner_cooling_rate
		if args.tuner_attempt_times is not None:
			manifest["tuner_config"]["attempt_times"] = args.tuner_attempt_times
		if args.tuner_mutate_count is not None:
			manifest["tuner_config"]["mutate_count"] = args.tuner_mutate_count
		if args.tuner_step_scale is not None:
			manifest["tuner_config"]["step_scale"] = args.tuner_step_scale
		if args.tuner_seed is not None:
			manifest["tuner_config"]["seed"] = args.tuner_seed
	if controller_log_path is not None:
		manifest["files"]["controller_log"] = str(controller_log_path)
		if tuner_enabled:
			manifest["files"]["tuner_log"] = str(run_dir / "tuner.log")
			manifest["files"]["monitor_output"] = str(run_dir / "monitor_output.tr")
	if args.scheme == "paraleon_plus":
		manifest["paraleon_plus_config"] = {
			"size_view": "wr_chunk",
			"chunk_kb": args.paraleon_plus_chunk_kb,
			"chunk_bytes": args.paraleon_plus_chunk_kb * 1024,
			"disable_potential_large": True,
		}
		manifest["files"]["host_flow_report"] = str(run_dir / "host_flow_report.tr")
	if args.scheme == "acc":
		manifest["files"]["acc_monitor"] = str(run_dir / "acc_monitor.tr")
		manifest["files"]["acc_dataset"] = str(run_dir / "acc_dataset.jsonl")
		manifest["files"]["acc_model"] = str(run_dir / "acc_model_final.npz")
		if acc_model_path is not None:
			manifest["files"]["acc_model_input"] = acc_model_path
	if args.scheme == "expert":
		manifest["expert_baseline"] = {
			"ai_rate_mbps": EXPERT_RATE_AI_MBPS,
			"hai_rate_mbps": EXPERT_RATE_HAI_MBPS,
			"rate_decrease_interval_us": EXPERT_RATE_DECREASE_INTERVAL_US,
			"min_time_between_cnps_us": EXPERT_MIN_TIME_BETWEEN_CNPS_US,
			"kmin_kb": EXPERT_KMIN_KB,
			"kmax_kb": EXPERT_KMAX_KB,
			"pmax": EXPERT_PMAX,
			"note": "min_time_between_cnps is not independently configurable in the current simulator path; expert scheme preserves the static 80us rate decrease interval and records the 96us paper value here for traceability.",
		}
	if args.scheme == "fixed_params":
		manifest["fixed_params_file"] = resolve_optional_path(args.fixed_params_file)
	write_json(manifest_path, manifest)

	env = os.environ.copy()
	env["PARALEON_RUN_DIR"] = str(run_dir.resolve())
	env["PARALEON_SCHEME"] = args.scheme
	if args.scheme == "expert":
		env["EXPERT_MIN_TIME_BETWEEN_CNPS"] = str(EXPERT_MIN_TIME_BETWEEN_CNPS_US)
	if args.collect_only:
		env["PARALEON_COLLECT_ONLY"] = "1"
	if args.aggressive_weights is not None:
		env["PARALEON_AGGRESSIVE_WEIGHTS"] = format_weight_triplet(args.aggressive_weights)
	if args.conservative_weights is not None:
		env["PARALEON_CONSERVATIVE_WEIGHTS"] = format_weight_triplet(args.conservative_weights)
	env["PARALEON_DIRECTION_PROB_CAP"] = "%.6g" % args.direction_prob_cap
	env["PARALEON_DIRECTION_CAP_POLICY"] = args.direction_cap_policy
	if args.aggressive_mode_large_flow_threshold is not None:
		env["PARALEON_AGGRESSIVE_MODE_LARGE_FLOW_THRESHOLD"] = "%.6g" % args.aggressive_mode_large_flow_threshold
	if args.tuner_initial_temperature is not None:
		env["PARALEON_TUNER_INITIAL_TEMPERATURE"] = "%.6g" % args.tuner_initial_temperature
	if args.tuner_final_temperature is not None:
		env["PARALEON_TUNER_FINAL_TEMPERATURE"] = "%.6g" % args.tuner_final_temperature
	if args.tuner_cooling_rate is not None:
		env["PARALEON_TUNER_COOLING_RATE"] = "%.6g" % args.tuner_cooling_rate
	if args.tuner_attempt_times is not None:
		env["PARALEON_TUNER_ATTEMPT_TIMES"] = str(args.tuner_attempt_times)
	if args.tuner_mutate_count is not None:
		env["PARALEON_TUNER_MUTATE_COUNT"] = str(args.tuner_mutate_count)
	if args.tuner_step_scale is not None:
		env["PARALEON_TUNER_STEP_SCALE"] = "%.6g" % args.tuner_step_scale
	if args.tuner_seed is not None:
		env["PARALEON_TUNER_SEED"] = str(args.tuner_seed)
	env["PARALEON_MONITOR_INTERVAL_MS"] = "%.6g" % args.monitor_interval_ms
	env["PARALEON_PARAMETER_TUNING_INTERVAL_MS"] = "%.6g" % args.parameter_tuning_interval_ms
	env["PARALEON_PLUS_CHUNK_BYTES"] = str(args.paraleon_plus_chunk_kb * 1024)
	env["PYTHONUNBUFFERED"] = "1"
	env["PYTHONDONTWRITEBYTECODE"] = "1"
	extend_simulator_library_path(env)

	print("Run directory:", run_dir)
	print("Topology:", topology_path)
	print("Flow file:", flow_path)
	print("Config:", config_path)

	controller_process = None
	controller_log = None
	simulator_process = None
	simulator_log = None
	simulator_returncode = None
	controller_returncode = None

	try:
		if controller_command is not None:
			controller_process, controller_log = launch_process(controller_command, controller_log_path, env)
			time.sleep(1)

		simulator_process, simulator_log = launch_process(resolve_simulator_command(config_path), run_dir / "simulator.log", env)
		simulator_returncode = simulator_process.wait()
	finally:
		if simulator_log is not None:
			simulator_log.close()
		controller_returncode = terminate_process(controller_process)
		if controller_log is not None:
			controller_log.close()
		move_runtime_artifacts(run_dir)

	manifest["completed_at"] = datetime.now().isoformat()
	manifest["simulator_returncode"] = simulator_returncode
	manifest["controller_returncode"] = controller_returncode
	if tuner_enabled:
		manifest["tuner_returncode"] = controller_returncode
	write_json(manifest_path, manifest)

	if simulator_returncode != 0:
		raise SystemExit("simulator failed, see %s" % (run_dir / "simulator.log"))

	print("Experiment complete.")
	print("Simulator log:", run_dir / "simulator.log")
	if controller_command is not None:
		print("Controller log:", controller_log_path)
	print("Manifest:", manifest_path)


if __name__ == "__main__":
	main()
