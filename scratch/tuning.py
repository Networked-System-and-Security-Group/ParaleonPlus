import math
import random
import numpy as np
import pandas as pd
import time
import hashlib
import json
from scipy.stats import entropy
import threading
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIX_DIR = os.environ.get('PARALEON_RUN_DIR', os.path.join(ROOT_DIR, 'mix'))
SCHEME = os.environ.get('PARALEON_SCHEME', 'paraleon')
USE_HOST_FLOW_REPORT = SCHEME == 'paraleon_plus'
TRACE_WRITE_LOCK = threading.Lock()
MONITOR_OUTPUT_HEADER = (
    'time,mode,throughput_raw,rtt_raw,pfc_raw,'
    'throughput_norm,rtt_norm,pfc_norm,'
    'throughput_utility,rtt_utility,pfc_utility,total_utility,'
    'large,small,total,large_ratio,small_ratio,'
    'large_count,potential_large,small_count,potential_effective,'
    'true_large_count,true_small_count,true_total_count,true_large_ratio,true_small_ratio,'
    'kl_divergence,trigger_reason,is_tuning'
)


def mix_path(file_name):
    return os.path.join(MIX_DIR, file_name)


def resolve_mode_weights(env_name, default_weights):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_weights
    return tuple(float(part.strip()) for part in raw_value.split(','))


def resolve_unit_interval(env_name, default_value):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    value = float(raw_value)
    if value < 0 or value > 1:
        raise ValueError(f'{env_name} must be between 0 and 1')
    return value


def resolve_positive_float(env_name, default_value):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f'{env_name} must be positive')
    return value


def resolve_open_unit_interval(env_name, default_value):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    value = float(raw_value)
    if value <= 0 or value >= 1:
        raise ValueError(f'{env_name} must be strictly between 0 and 1')
    return value


def resolve_positive_int(env_name, default_value):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f'{env_name} must be positive')
    return value


def resolve_ratio_threshold(env_name, default_value):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value
    value = float(raw_value)
    if value < 0 or value > 1:
        raise ValueError(f'{env_name} must be between 0 and 1')
    return value


def resolve_positive_ms(env_name, default_value_ms):
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default_value_ms
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f'{env_name} must be positive')
    return value


def load_total_directed_links():
    manifest_path = mix_path('manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as manifest_file:
            manifest = json.load(manifest_file)
        link_count = manifest.get('topology', {}).get('link_count')
        if link_count:
            return int(link_count) * 2

    config_path = mix_path('config.txt')
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            for line in config_file:
                if line.startswith('TOPOLOGY_FILE '):
                    topology_file = line.split(maxsplit=1)[1].strip()
                    if not os.path.isabs(topology_file):
                        topology_file = os.path.join(MIX_DIR, topology_file)
                    with open(topology_file, 'r') as topology_handle:
                        header = topology_handle.readline().split()
                    if len(header) == 3:
                        return int(header[2]) * 2

    return 1


def normalize_log_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [normalize_log_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_log_value(item) for key, item in value.items()}
    return value


def solution_to_parameter_map(parameter_values):
    return {
        spec['name']: normalize_log_value(value)
        for spec, value in zip(PARAMETER_SPECS, parameter_values)
    }


def format_log_value(value, digits=6):
    if isinstance(value, (np.bool_, bool)):
        return 'true' if bool(value) else 'false'
    if value is None:
        return 'n/a'
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return 'nan'
        if math.isinf(numeric_value):
            return 'inf' if numeric_value > 0 else '-inf'
        return f'{numeric_value:.{digits}f}'.rstrip('0').rstrip('.')
    return str(value)


def calculate_utility_breakdown(throughput, rtt, pfc, throughput_weight, rtt_weight, pfc_weight):
    throughput_norm = throughput / base_throughput if base_throughput else 0
    rtt_norm = base_rtt / rtt if rtt > 0 else 0
    # pfc_norm = 1 - pfc * pfc_pause_time / t_tune if t_tune else 0
    pfc_norm = max(0, 1 - pfc)

    throughput_utility = throughput_weight * throughput_norm
    rtt_utility = rtt_weight * rtt_norm
    pfc_utility = pfc_weight * pfc_norm

    return {
        'throughput': throughput,
        'rtt': rtt,
        'pfc': pfc,
        'throughput_norm': throughput_norm,
        'rtt_norm': rtt_norm,
        'pfc_norm': pfc_norm,
        'throughput_weight': throughput_weight,
        'rtt_weight': rtt_weight,
        'pfc_weight': pfc_weight,
        'throughput_utility': throughput_utility,
        'rtt_utility': rtt_utility,
        'pfc_utility': pfc_utility,
        'total_utility': throughput_utility + rtt_utility + pfc_utility,
    }


def write_trace_block(sim_time, detail_lines):
    block_lines = [f'sim_time={format_log_value(sim_time)}']
    block_lines.extend('\t' + line for line in detail_lines)
    block = '\n'.join(block_lines)

    with TRACE_WRITE_LOCK:
        print(block, flush=True)
        print(flush=True)


def latest_snapshot_time(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None

    latest_time = None
    with open(file_path, 'r') as file_handle:
        for line in file_handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                latest_time = float(parts[0])
            except ValueError:
                continue
    return latest_time


def wait_for_monitor_snapshots_at_or_after(target_time):
    while True:
        throughput_time = latest_snapshot_time(throughput_input_file)
        rtt_time = latest_snapshot_time(rtt_input_file)
        if throughput_time is not None and rtt_time is not None:
            if throughput_time >= target_time and rtt_time >= target_time:
                return
        time.sleep(0.05)


def format_flow_distribution_line(flow_ratio, previous_flow_ratio=None, kl_divergence=None):
    large_flow = float(flow_ratio.get('large', 0))
    small_flow = float(flow_ratio.get('small', 0))
    total_flow = large_flow + small_flow

    if total_flow > 0:
        large_ratio = large_flow / total_flow
        small_ratio = small_flow / total_flow
    else:
        large_ratio = 0
        small_ratio = 0

    parts = [
        'flow_distribution: '
        + f'large={format_log_value(large_flow)}, '
        + f'small={format_log_value(small_flow)}, '
        + f'total={format_log_value(total_flow)}, '
        + f'large_ratio={format_log_value(large_ratio)}, '
        + f'small_ratio={format_log_value(small_ratio)}'
    ]

    observed_large = flow_ratio.get('large_count')
    potential_large = flow_ratio.get('potential_large')
    observed_small = flow_ratio.get('small_count')
    potential_effective = flow_ratio.get('potential_effective')
    if observed_large is not None or potential_large is not None or observed_small is not None:
        detail_parts = []
        if observed_large is not None:
            detail_parts.append(f"large={format_log_value(observed_large)}")
        if potential_large is not None:
            detail_parts.append(f"potential_large={format_log_value(potential_large)}")
        if observed_small is not None:
            detail_parts.append(f"small={format_log_value(observed_small)}")
        if potential_effective is not None:
            detail_parts.append(f"potential_effective={format_log_value(potential_effective)}")
        parts.append('observed_distribution: ' + ', '.join(detail_parts))

    if previous_flow_ratio is not None and previous_flow_ratio.get('small', -1) != -1:
        previous_large = float(previous_flow_ratio.get('large', 0))
        previous_small = float(previous_flow_ratio.get('small', 0))
        previous_total = previous_large + previous_small

        if previous_total > 0:
            previous_large_ratio = previous_large / previous_total
            previous_small_ratio = previous_small / previous_total
        else:
            previous_large_ratio = 0
            previous_small_ratio = 0

        parts.append(
            'previous_distribution: '
            + f'large_ratio={format_log_value(previous_large_ratio)}, '
            + f'small_ratio={format_log_value(previous_small_ratio)}'
        )

    if kl_divergence is not None:
        parts.append(f'kl_divergence={format_log_value(kl_divergence)}')

    return ', '.join(parts)


def format_utility_line(utility_breakdown):
    return (
        'utility: '
        + f"throughput_norm={format_log_value(utility_breakdown['throughput_norm'])} "
        + f"(raw={format_log_value(utility_breakdown['throughput'])}, weighted={format_log_value(utility_breakdown['throughput_utility'])}), "
        + f"rtt_norm={format_log_value(utility_breakdown['rtt_norm'])} "
        + f"(raw={format_log_value(utility_breakdown['rtt'])}, weighted={format_log_value(utility_breakdown['rtt_utility'])}), "
        + f"pfc_norm={format_log_value(utility_breakdown['pfc_norm'])} "
        + f"(raw={format_log_value(utility_breakdown['pfc'])}, weighted={format_log_value(utility_breakdown['pfc_utility'])}), "
        + f"total={format_log_value(utility_breakdown['total_utility'])}"
    )


def format_parameter_line(parameter_values):
    parameter_map = solution_to_parameter_map(parameter_values)
    parameter_items = [
        f"{spec['name']}={format_log_value(parameter_map[spec['name']])}"
        for spec in PARAMETER_SPECS
    ]
    return 'parameters: ' + ', '.join(parameter_items)


def format_tuning_line(tuning_info):
    if not tuning_info:
        return 'tuning: state=idle'

    ordered_fields = [
        ('state', 'state'),
        ('mode', 'mode'),
        ('trigger_reason', 'trigger'),
        ('tuning_round_index', 'session'),
        ('tuning_round', 'round'),
        ('temperature', 'temperature'),
        ('candidate_type', 'candidate'),
        ('accepted', 'accepted'),
        ('best_value', 'best_utility'),
    ]
    field_parts = []

    for key, label in ordered_fields:
        if key not in tuning_info or tuning_info[key] is None:
            continue
        field_parts.append(f'{label}={format_log_value(tuning_info[key])}')

    if not field_parts:
        field_parts.append('state=idle')

    return 'tuning: ' + ', '.join(field_parts)


def compute_flow_distribution_stats(flow_ratio):
    large_flow = float(flow_ratio.get('large', 0))
    small_flow = float(flow_ratio.get('small', 0))
    total_flow = large_flow + small_flow
    if total_flow > 0:
        large_ratio = large_flow / total_flow
        small_ratio = small_flow / total_flow
    else:
        large_ratio = 0
        small_ratio = 0
    return {
        'large': large_flow,
        'small': small_flow,
        'total': total_flow,
        'large_ratio': large_ratio,
        'small_ratio': small_ratio,
    }


def compute_count_ratio_stats(large_count, small_count):
    if large_count is None or small_count is None:
        return {
            'large_count': None,
            'small_count': None,
            'total_count': None,
            'large_ratio': None,
            'small_ratio': None,
        }

    large_count = float(large_count)
    small_count = float(small_count)
    total_count = large_count + small_count
    if total_count > 0:
        large_ratio = large_count / total_count
        small_ratio = small_count / total_count
    else:
        large_ratio = 0
        small_ratio = 0
    return {
        'large_count': large_count,
        'small_count': small_count,
        'total_count': total_count,
        'large_ratio': large_ratio,
        'small_ratio': small_ratio,
    }


def format_csv_value(value):
    if value is None:
        return ''
    return format_log_value(value)


def write_monitor_output_row(sim_time, parameter_mode, utility_breakdown, flow_ratio,
                             flow_ratio_metadata, kl_divergence, trigger_reason, tuning_active):
    stats = compute_flow_distribution_stats(flow_ratio)
    metadata = {
        'large_count': flow_ratio_metadata.get('large_count', int(stats['large'])),
        'potential_large': flow_ratio_metadata.get('potential_large', 0),
        'small_count': flow_ratio_metadata.get('small_count', int(stats['small'])),
        'potential_effective': flow_ratio_metadata.get('potential_effective', 0),
    }
    true_stats = compute_count_ratio_stats(
        flow_ratio_metadata.get('true_large_count'),
        flow_ratio_metadata.get('true_small_count'),
    )
    row = [
        format_log_value(sim_time),
        str(parameter_mode),
        format_log_value(utility_breakdown['throughput']),
        format_log_value(utility_breakdown['rtt']),
        format_log_value(utility_breakdown['pfc']),
        format_log_value(utility_breakdown['throughput_norm']),
        format_log_value(utility_breakdown['rtt_norm']),
        format_log_value(utility_breakdown['pfc_norm']),
        format_log_value(utility_breakdown['throughput_utility']),
        format_log_value(utility_breakdown['rtt_utility']),
        format_log_value(utility_breakdown['pfc_utility']),
        format_log_value(utility_breakdown['total_utility']),
        format_log_value(stats['large']),
        format_log_value(stats['small']),
        format_log_value(stats['total']),
        format_log_value(stats['large_ratio']),
        format_log_value(stats['small_ratio']),
        format_log_value(metadata['large_count']),
        format_log_value(metadata['potential_large']),
        format_log_value(metadata['small_count']),
        format_log_value(metadata['potential_effective']),
        format_csv_value(true_stats['large_count']),
        format_csv_value(true_stats['small_count']),
        format_csv_value(true_stats['total_count']),
        format_csv_value(true_stats['large_ratio']),
        format_csv_value(true_stats['small_ratio']),
        format_log_value(kl_divergence),
        '' if trigger_reason is None else str(trigger_reason),
        'true' if tuning_active else 'false',
    ]

    with TRACE_WRITE_LOCK:
        file_exists = os.path.exists(monitor_output_file)
        with open(monitor_output_file, 'a') as file_handle:
            if not file_exists or os.path.getsize(monitor_output_file) == 0:
                print(MONITOR_OUTPUT_HEADER, file=file_handle)
            print(','.join(row), file=file_handle)


def log_cycle_summary(sim_time, flow_ratio, utility_breakdown, parameter_values, tuning_info=None,
                      previous_flow_ratio=None, kl_divergence=None):
    write_trace_block(
        sim_time,
        [
            format_flow_distribution_line(flow_ratio, previous_flow_ratio, kl_divergence),
            format_utility_line(utility_breakdown),
            format_parameter_line(parameter_values),
            format_tuning_line(tuning_info),
        ],
    )


def collect_observation_metrics(current_time):
    wait_for_monitor_snapshots_at_or_after(current_time)
    throughput_matrix = throughput_handling2()
    rtt_matrix = rtt_handling2()
    avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
    avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
    start_time_this_round = current_time - t_tune
    avg_pfc = pfc_handling(start_time_this_round, current_time, pfc_start_line)
    return avg_throughput, avg_rtt, avg_pfc


def log_monitor_observation(current_time, flow_ratio_snapshot, previous_flow_ratio_snapshot,
                            current_parameter_values, kl_divergence):
    avg_throughput, avg_rtt, avg_pfc = collect_observation_metrics(current_time)
    parameter_mode, throughput_weight, rtt_weight, pfc_weight = judge_mode(
        flow_ratio_snapshot['large'],
        flow_ratio_snapshot['small'],
    )
    utility_breakdown = calculate_utility_breakdown(
        avg_throughput,
        avg_rtt,
        avg_pfc,
        throughput_weight,
        rtt_weight,
        pfc_weight,
    )
    output_metric(current_time, avg_throughput, avg_rtt, avg_pfc, utility_breakdown['total_utility'])
    log_cycle_summary(
        current_time,
        flow_ratio_snapshot,
        utility_breakdown,
        current_parameter_values,
        {
            'state': 'observe',
            'mode': parameter_mode,
            'candidate_type': 'current',
        },
        previous_flow_ratio_snapshot,
        kl_divergence,
    )


def calculate_file_hash(file_path):
    if not os.path.exists(file_path):
        return None

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()

def potential_weight(observed_size, threshold_size):
    if threshold_size <= 0:
        return 0
    if observed_size <= 0:
        return 0
    return min(float(observed_size) / float(threshold_size), 1.0)


def is_strictly_increasing(values):
    if len(values) < 2:
        return False
    for index in range(1, len(values)):
        if values[index] <= values[index - 1]:
            return False
    return True

def is_flow_keep_sending_sliding_window(lst, window_size):
    result = []
    for i in range(len(lst) - window_size + 1):
        window_sum = sum(lst[i:i+window_size])
        result.append(window_sum)

    for i in range(1, len(result)-1):
        if result[i] == 0:
            return False
        elif result[i-1] - result[i] > 10:
            if result[i] - result[i + 1] > 20:
                return False
    return True

def is_flow_keep_active_within_window(lst, window_size):
    if len(lst) <= window_size:
        return False
    
    result = []
    for i in range(len(lst) - window_size + 1):
        window_sum = sum(lst[i:i+window_size])
        result.append(window_sum)

    for i in range(1, len(result)-1):
        if result[i] == 0:
            return False
        elif result[i-1] - result[i] > 10:
            if len(result) < 3:
                continue
            if result[i] - result[i + 1] > 20:
                return False
    return True


def update_all_flows(current_flowid_size_dict, current_time):
    global switch_flowid_status_dict
    
    for switch_id, flowid_size_dict in current_flowid_size_dict.items():
        if switch_id not in switch_flowid_status_dict:
            switch_flowid_status_dict[switch_id] = {}
        
        for flow_id, flow_size in flowid_size_dict.items():
            if flow_id not in switch_flowid_status_dict[switch_id]:
                temp_flowid_status = {'active': True, 'flow_size': [], 'last_monitor_time': current_time, 'potential_start': -1, 'potential_stop': -1}
                switch_flowid_status_dict[switch_id][flow_id] = temp_flowid_status
            
            switch_flowid_status_dict[switch_id][flow_id]['flow_size'].append(flow_size)
            switch_flowid_status_dict[switch_id][flow_id]['last_monitor_time'] = current_time
            switch_flowid_status_dict[switch_id][flow_id]['active'] = True

    for switch_id, flow_dataset in switch_flowid_status_dict.items():
        for flow_id, flow_status in flow_dataset.items():
            if flow_status['active'] == False:
                continue
            if current_time - flow_status['last_monitor_time'] > t_reset / 2:
                flow_status['active'] = False
                continue


def filter_flows():
    global switch_flowid_status_dict
    switch_upload_dict = {}

    for switch_id, single_flowid_status_dict in switch_flowid_status_dict.items():
        if switch_id not in switch_upload_dict:
            flow_type_list_dict = {'large': [], 'potential_large': {}, 'small': []}
            switch_upload_dict[switch_id] = flow_type_list_dict
        
        for flow_id, flow_status in single_flowid_status_dict.items():
            flow_active = flow_status['active']
            flow_size = sum(flow_status['flow_size'])
            
            if flow_active:
                if flow_size >= large_flow_threshold:
                    switch_upload_dict[switch_id]['large'].append(flow_id)
                    if flow_id in switch_upload_dict[switch_id]['potential_large']:
                        del switch_upload_dict[switch_id]['potential_large'][flow_id]
                        flow_status['potential_stop'] = current_time

                else:
                    if is_flow_keep_active_within_window(flow_status['flow_size'], window_size):
                        switch_upload_dict[switch_id]['potential_large'][flow_id] = flow_size
                        if flow_status['potential_start'] == -1:
                            flow_status['potential_start'] = current_time

                        if flow_id in switch_upload_dict[switch_id]['small']:
                            switch_upload_dict[switch_id]['small'].remove(flow_id)
                    else:
                        switch_upload_dict[switch_id]['small'].append(flow_id)
            else:
                if flow_id in switch_upload_dict[switch_id]['large']:
                    switch_upload_dict[switch_id]['large'].remove(flow_id)

                if flow_id in switch_upload_dict[switch_id]['potential_large']:
                    del switch_upload_dict[switch_id]['potential_large'][flow_id]
                    if flow_status['potential_stop'] == -1:
                        flow_status['potential_stop'] = current_time

                if flow_id in switch_upload_dict[switch_id]['small']:
                    switch_upload_dict[switch_id]['small'].remove(flow_id)
    
    return switch_upload_dict


def load_sketch():
    if not os.path.exists(sketch_heavypart_path) or os.path.getsize(sketch_heavypart_path) == 0:
        return None, {}

    file_column = ['time', 'switch']
    current_flowid_size_dict = {}

    for i in range(10000):
        file_column.append('p' + str(i))

    try:
        data = pd.read_table(sketch_heavypart_path, names=file_column, header=None, sep='\s+', low_memory=False).fillna(value='-1na')
    except pd.errors.EmptyDataError:
        return None, {}

    if data.empty:
        return None, {}

    current_time = None

    for index, row in data.iterrows():
        current_time = row.iloc[0]
        switch_id = row.iloc[1]

        if switch_id not in current_flowid_size_dict:
            current_flowid_size_dict[switch_id] = {}

        if row.shape[0] > 1:
            for i in range(2, row.shape[0]):
                if 'na' not in row.iloc[i]:
                    flowid_size_pair = row.iloc[i]
                    flow_id = flowid_size_pair.split('-')[0]
                    flow_size = int(flowid_size_pair.split('-')[1])
                    current_flowid_size_dict[switch_id][flow_id] = flow_size
                else:
                    break

    return current_time, current_flowid_size_dict


def load_host_flow_report():
    if not os.path.exists(host_flow_report_path) or os.path.getsize(host_flow_report_path) == 0:
        return None, {}

    current_time = None
    current_flow_progress = {}

    with open(host_flow_report_path, 'r') as report_file:
        for line in report_file:
            parts = line.strip().split()
            if len(parts) == 5 and parts[1] == 'wr_summary':
                current_time = float(parts[0])
                continue
            if len(parts) != 7 or parts[1] != 'wr':
                continue

            current_time = float(parts[0])
            host_id = parts[2]
            qpn_key = parts[4]
            flow_id = f'{host_id}-{qpn_key}'
            current_flow_progress[flow_id] = {
                'progress_bytes': int(parts[5]),
                'flow_size': int(parts[6]),
            }

    return current_time, current_flow_progress


def compute_chunk_visible_size(progress_bytes, flow_size):
    if progress_bytes <= 0:
        return 0
    aligned_progress = ((progress_bytes + HOST_FLOW_CHUNK_BYTES - 1) // HOST_FLOW_CHUNK_BYTES) * HOST_FLOW_CHUNK_BYTES
    return min(flow_size, aligned_progress)


def classify_host_flow_progress(current_flow_progress_dict):
    current_ratio = {'large': 0, 'small': 0}
    metadata = {
        'large_count': 0,
        'potential_large': 0,
        'small_count': 0,
        'potential_effective': 0,
        'true_large_count': 0,
        'true_small_count': 0,
    }

    for flow_progress in current_flow_progress_dict.values():
        if flow_progress['flow_size'] >= large_flow_threshold_bytes:
            metadata['true_large_count'] += 1
        else:
            metadata['true_small_count'] += 1
        visible_size = compute_chunk_visible_size(flow_progress['progress_bytes'], flow_progress['flow_size'])

        if visible_size >= large_flow_threshold_bytes:
            metadata['large_count'] += 1
            current_ratio['large'] += 1
            continue

        metadata['small_count'] += 1
        current_ratio['small'] += 1
    return current_ratio, metadata


def upload_ratio(switch_upload_dict, current_time):
    global current_flow_ratio
    metadata = {'large_count': 0, 'potential_large': 0, 'small_count': 0, 'potential_effective': 0}
    for switch_id, flow_type_list_dict in switch_upload_dict.items():
        for flow_type, flow_type_list in flow_type_list_dict.items():
            if flow_type == 'large':
                current_flow_ratio['large'] += len(flow_type_list)
                metadata['large_count'] += len(flow_type_list)
            elif flow_type == 'small':
                current_flow_ratio['small'] += len(flow_type_list)
                metadata['small_count'] += len(flow_type_list)
            elif flow_type == 'potential_large':
                for flow_size in flow_type_list.values():
                    effective_weight = potential_weight(flow_size, large_flow_threshold)
                    current_flow_ratio['large'] += effective_weight
                    metadata['potential_large'] += 1
                    metadata['potential_effective'] += effective_weight
    return metadata


def compute_divergence(current_flow_ratio, previous_flow_ratio):
    current_total = sum(current_flow_ratio.values())
    previous_total = sum(previous_flow_ratio.values())

    if current_total <= 0 or previous_total <= 0:
        return 0

    current_prob = np.array(list(current_flow_ratio.values())) / current_total
    previous_prob = np.array(list(previous_flow_ratio.values())) / previous_total
    kl_divergence = entropy(current_prob, previous_prob)

    return kl_divergence


def mean_or_default(values, default_value):
    if values.size == 0:
        return default_value
    return np.mean(values)

def integer_candidates(start, stop):
    return list(range(start, stop))


def decimal_candidates(start, stop, step):
    return [round(value, 6) for value in np.arange(start, stop, step)]


def default_parameter_values():
    return normalize_parameter_values([spec['default'] for spec in PARAMETER_SPECS])


# One human-edited source of truth for tunable parameters.
ALL_PARAMETER_SPECS = [
    # Rate increase.
    {
        'group': 'rate_increase',
        'name': 'time_reset',
        'values': integer_candidates(50, 600),# 管理从AI到HAI的跳转时间，从[1, 1200)改为[50, 600)
        'default': 300,
        'type': 'int',
        'aggressive_direction': -1,
        'min': 50,
        'max': 600,
        'step': 50,
    },
    {
        'group': 'rate_increase',
        'name': 'ai_rate',
        'values': integer_candidates(5, 100),# 从[1, 500)改为[2, 80)
        'default': 5,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 5,
        'max': 100,
        'step': 5,
    },
    {
        'group': 'rate_increase',
        'name': 'hai_rate',
        'values': integer_candidates(30, 600),# 从[10, 5000)改为[30, 600)
        'default': 50,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 30,
        'max': 600,
        'step': 50,
    },

    # Rate decrease.
    {
        'group': 'rate_decrease',
        'name': 'rate_to_set_on_first_cnp',
        'values': decimal_candidates(0.3, 0.8, 0.1), # 从[0.1, 1.0)改为[0.3, 0.8)
        'default': 0.5,
        'type': 'float',
        'aggressive_direction': 1,
        'min': 0.3,
        'max': 0.8,
        'step': 0.1,
    },
    {
        'group': 'rate_decrease',
        'name': 'rpg_min_dec_fac',#单次最大降幅
        'values': decimal_candidates(0.3, 0.8, 0.1), # 从[0.1, 1.0)改为[0.3, 0.8)
        'default': 0.5,
        'type': 'float',
        'aggressive_direction': -1,
        'min': 0.3,
        'max': 0.8,
        'step': 0.1,
    },
    {
        'group': 'rate_decrease',
        'name': 'rpg_min_rate', #最小速率，单位Mb/s
        'values': integer_candidates(1, 5000), 
        'default': 1,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 1,
        'max': 5000,
        'step': 300,
    },
    {
        'group': 'rate_decrease',
        'name': 'rpg_gd',  #coefficient between alpha and rate reduction factor
        'values': integer_candidates(11, 13), # 尝试将其从从1-13改为[11, 13)
        'default': 11, # Rc = Rc(1-alpha*2^(10-rpg_gd))
        'type': 'int',
        'aggressive_direction': 1,
        'min': 11,
        'max': 13,
        'step': 1,
    },
    {
        'group': 'rate_decrease',
        'name': 'min_time_between_cnps',   # 邻近的两次降速最短时间
        'values': integer_candidates(4, 128), # 从[0, 4065)改为[4, 128)
        'default': 4,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 4,
        'max': 128,
        'step': 8,
    },

    # Alpha update.
    {
        'group': 'alpha_update',
        'name': 'dce_tcp_g',
        'values': integer_candidates(960, 1020),  # 除以1024就是1-g 从(0, 1019)改到[960, 1020)
        'default': 1019,
        'type': 'int',
        'aggressive_direction': -1,
        'min': 960,
        'max': 1020,
        'step': 5,
    },
    {
        'group': 'alpha_update',
        'name': 'initial_alpha_value', # 疑似无用
        'values': integer_candidates(0, 1023),
        'default': 1023,
        'type': 'int',
        'aggressive_direction': -1,
        'min': 1,
        'max': 1023,
        'step': 16,
    },

    # ECN threshold.
    {
        'group': 'ecn_threshold',
        'name': 'kmin',
        'values': integer_candidates(40, 1600),
        'default': 400,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 40,
        'max': 1600,
        'step': 100,
    },
    {
        'group': 'ecn_threshold',
        'name': 'kmax',
        'values': integer_candidates(480, 6400),
        'default': 1600,
        'type': 'int',
        'aggressive_direction': 1,
        'min': 480,
        'max': 6400,
        'step': 400,
    },
    {
        'group': 'ecn_threshold',
        'name': 'pmax',
        'values': decimal_candidates(0.1, 0.4, 0.01),
        'default': 0.2,
        'type': 'float',
        'aggressive_direction': -1,
        'min': 0.1,
        'max': 0.4,
        'step': 0.01,
    },
]

PARAMETER_SPECS = ALL_PARAMETER_SPECS
PARAMETER_SPEC_BY_NAME = {spec['name']: spec for spec in PARAMETER_SPECS}
PARAMETER_INDEX_BY_NAME = {spec['name']: index for index, spec in enumerate(PARAMETER_SPECS)}


def normalize_parameter_values(parameter_values):
    normalized = list(parameter_values)
    kmin_index = PARAMETER_INDEX_BY_NAME['kmin']
    kmax_index = PARAMETER_INDEX_BY_NAME['kmax']
    if normalized[kmin_index] > normalized[kmax_index]:
        normalized[kmin_index], normalized[kmax_index] = normalized[kmax_index], normalized[kmin_index]
    return normalized


def utility_function(throughput, rtt, pfc, throughput_weight, rtt_weight, pfc_weight):
    utility_breakdown = calculate_utility_breakdown(throughput, rtt, pfc, throughput_weight, rtt_weight, pfc_weight)
    return utility_breakdown['total_utility']


def rtt_handling2():
    data = pd.read_table(rtt_input_file, header=None, sep='\s+', names=['time', 'sip-dip', 'rtt'], low_memory=False).dropna()
    if data.empty:
        return np.array([])

    rtt_values = []
    for index, row in data.iterrows():
        sip_dip_pair = row.iloc[1]
        rtt_value = row.iloc[2] / 1000
        if '-' not in str(sip_dip_pair):
            continue
        rtt_values.append(rtt_value)

    return np.array(rtt_values)
    

def throughput_handling2():
    data = pd.read_table(throughput_input_file, header=None, sep='\s+', names=['time', 'switch', 'port', 'rxBytes'], low_memory=False).dropna()
    if data.empty:
        return np.array([])

    throughput_values = []

    for index, row in data.iterrows():
        port_id = row.iloc[2]
        rx_bytes = row.iloc[3]

        if port_id <= 0:
            continue

        throughput_values.append(rx_bytes * 8 / monitor_window_seconds / 1e9)

    return np.array(throughput_values)

def pfc_handling_with_state(start_time_this_round, stop_time_this_round, start_line, active_links_state):
    active_links_this_round = set(active_links_state)

    data = pd.read_table(
        pfc_input_file,
        header=None,
        sep='\s+',
        names=['time', 'node_id', 'node_type', 'if_index', 'pfc_type'],
        low_memory=False,
    ).dropna()

    next_start_line = start_line
    next_active_links_state = set(active_links_state)

    for index, row in data[start_line:].iterrows():
        current_time = row.iloc[0]
        node_id = int(row.iloc[1])
        if_index = int(row.iloc[3])
        pfc_type = int(row.iloc[-1])
        link_key = (node_id, if_index)

        if current_time < start_time_this_round:
            if pfc_type == 20000:
                next_active_links_state.add(link_key)
            elif pfc_type == 10000:
                next_active_links_state.discard(link_key)
            continue
        elif start_time_this_round <= current_time <= stop_time_this_round:
            if link_key in next_active_links_state:
                active_links_this_round.add(link_key)
            if pfc_type == 20000:
                next_active_links_state.add(link_key)
                active_links_this_round.add(link_key)
            elif pfc_type == 10000:
                next_active_links_state.discard(link_key)
        else:
            next_start_line = index
            break
    else:
        next_start_line = len(data)

    active_ratio = len(active_links_this_round) / total_directed_links if total_directed_links else 0
    return active_ratio, next_start_line, next_active_links_state


def pfc_handling(start_time_this_round, stop_time_this_round, start_line):
    global pfc_start_line, pfc_active_links
    active_ratio, pfc_start_line, pfc_active_links = pfc_handling_with_state(
        start_time_this_round,
        stop_time_this_round,
        start_line,
        pfc_active_links,
    )
    return active_ratio


def collect_monitor_cycle_summary(current_time, flow_ratio_snapshot, flow_ratio_metadata,
                                  previous_flow_ratio_snapshot, kl_divergence, trigger_reason):
    global observation_pfc_start_line, observation_pfc_active_links

    wait_for_monitor_snapshots_at_or_after(current_time)

    throughput_matrix = throughput_handling2()
    rtt_matrix = rtt_handling2()
    start_time_this_round = max(0, current_time - monitor_window_seconds)
    avg_pfc, observation_pfc_start_line, observation_pfc_active_links = pfc_handling_with_state(
        start_time_this_round,
        current_time,
        observation_pfc_start_line,
        observation_pfc_active_links,
    )

    avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
    avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
    parameter_mode, throughput_weight, rtt_weight, pfc_weight = judge_mode(
        flow_ratio_snapshot['large'],
        flow_ratio_snapshot['small'],
    )
    utility_breakdown = calculate_utility_breakdown(
        avg_throughput,
        avg_rtt,
        avg_pfc,
        throughput_weight,
        rtt_weight,
        pfc_weight,
    )
    parameter_values = load_current_parameters()
    tuning_info = {
        'state': 'observe',
        'mode': parameter_mode,
        'trigger_reason': trigger_reason,
    }
    log_cycle_summary(
        current_time,
        flow_ratio_snapshot,
        utility_breakdown,
        parameter_values,
        tuning_info,
        previous_flow_ratio_snapshot,
        kl_divergence,
    )
    write_monitor_output_row(
        current_time,
        parameter_mode,
        utility_breakdown,
        flow_ratio_snapshot,
        flow_ratio_metadata,
        kl_divergence,
        trigger_reason,
        is_tuning,
    )
    return utility_breakdown


def get_new_solution_direction(parameter_mode):
    global current_flow_ratio, previous_flow_ratio

    if current_flow_ratio['large'] != 0 and current_flow_ratio['small'] != 0:
        large_flow_number = current_flow_ratio['large']
        small_flow_number = current_flow_ratio['small']
    else:
        large_flow_number = previous_flow_ratio['large']
        small_flow_number = previous_flow_ratio['small']
    
    large_flow_ratio = large_flow_number / (large_flow_number + small_flow_number)
    small_flow_ratio = small_flow_number / (large_flow_number + small_flow_number)
    random_number = random.random()
    direction_prob_cap = get_direction_prob_cap()

    if parameter_mode == 'aggressive':
        if random_number <= min(large_flow_ratio, direction_prob_cap):
            return 1
        else:
            return -1
    else:
        if random_number <= min(small_flow_ratio, direction_prob_cap):
            return 1
        else:
            return -1


def generate_new_parameters(parameter_mode, current_solution):
    new_solution = []
    if parameter_mode == 'default':
        return default_parameter_values()

    mutate_count = min(MUTATE_COUNT, len(PARAMETER_SPECS))
    if mutate_count == 0:
        return normalize_parameter_values(current_solution)
    mutate_indices = set(random.sample(range(len(PARAMETER_SPECS)), mutate_count))
    for parameter_index, spec in enumerate(PARAMETER_SPECS):
        if parameter_index not in mutate_indices:
            new_solution.append(current_solution[parameter_index])
            continue

        direction = get_new_solution_direction(parameter_mode)
        mode_sign = spec['aggressive_direction'] if parameter_mode == 'aggressive' else -spec['aggressive_direction']
        parameter_value = (
            current_solution[parameter_index]
            + mode_sign * direction * spec['step'] * STEP_SCALE * random.uniform(0.5, 1)
        )

        if spec['type'] == 'int':
            parameter_value = round(parameter_value)
        else:
            parameter_value = 1 if parameter_value > 1 else parameter_value

        if parameter_value > spec['max']:
            parameter_value = spec['max']

        if parameter_value < spec['min']:
            parameter_value = spec['min']

        new_solution.append(parameter_value)

    return normalize_parameter_values(new_solution)


def get_direction_prob_cap():
    if DIRECTION_CAP_POLICY == 'fixed':
        return DIRECTION_PROB_CAP_MAX
    return DIRECTION_PROB_CAP_MIN + (DIRECTION_PROB_CAP_MAX - DIRECTION_PROB_CAP_MIN) * (1 - bad_rate_ewma)


def update_bad_rate_ewma(delta):
    global bad_rate_ewma
    bad_signal = 1.0 if delta < -DIRECTION_DELTA_EPS else 0.0
    bad_rate_ewma = DIRECTION_BAD_EWMA_ALPHA * bad_signal + (1 - DIRECTION_BAD_EWMA_ALPHA) * bad_rate_ewma

def judge_mode(large_flow_num, small_flow_num):
    total_flow_num = large_flow_num + small_flow_num
    large_flow_ratio = (large_flow_num / total_flow_num) if total_flow_num > 0 else 0
    if large_flow_ratio >= AGGRESSIVE_MODE_LARGE_FLOW_THRESHOLD:
        throughput_weight, rtt_weight, pfc_weight = AGGRESSIVE_WEIGHTS
        return 'aggressive', throughput_weight, rtt_weight, pfc_weight
    else:
        throughput_weight, rtt_weight, pfc_weight = CONSERVATIVE_WEIGHTS
        return 'conservative', throughput_weight, rtt_weight, pfc_weight


def load_current_parameters():
    if not os.path.exists(parameter_output_file) or os.path.getsize(parameter_output_file) == 0:
        return default_parameter_values()

    parameter_map = {}
    with open(parameter_output_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                key, value = line.strip().split('=')
                spec = PARAMETER_SPEC_BY_NAME.get(key)
                if spec is None:
                    continue
                if spec['type'] == 'float':
                    parameter_map[key] = float(value.strip())
                else:
                    parameter_map[key] = int(value.strip())

    return normalize_parameter_values([
        parameter_map.get(spec['name'], spec['default']) for spec in PARAMETER_SPECS
    ])


def write_parameters(parameter_values):
    parameter_values = normalize_parameter_values(parameter_values)
    with open(parameter_output_file, 'w') as f:
        for spec, value in zip(PARAMETER_SPECS, parameter_values):
            output_parameter_str = spec['name'] + '=' + str(value)
            print(output_parameter_str, file=f)

def output_metric(current_time, avg_throughput, avg_rtt, avg_pfc, utility_value):
    with open(metric_output_file, 'a') as f:
        output_str = str(current_time) + ' ' + str(avg_throughput) + ' ' + str(avg_rtt) + ' ' + str(avg_pfc) + ' ' + str(utility_value)
        print(output_str, file=f)


def wait_for_throughput_snapshot_after(target_time):
    while True:
        temp_throughput_data = pd.read_table(
            throughput_input_file,
            header=None,
            sep='\s+',
            names=['time', 'switch', 'port', 'rxBytes'],
            low_memory=False,
        ).dropna()
        if temp_throughput_data.empty:
            time.sleep(0.5)
            continue

        latest_time = temp_throughput_data.iloc[-1, 0]
        if latest_time > target_time:
            return latest_time

        time.sleep(0.5)


def collect_metrics_only(parameter_mode, throughput_weight, rtt_weight, pfc_weight, current_time,
                         flow_ratio_snapshot, previous_flow_ratio_snapshot, kl_divergence,
                         trigger_reason, tuning_round_index):
    start_time_this_round = current_time - t_tune
    stop_time_this_round = current_time

    current_solution = load_current_parameters()
    throughput_matrix = throughput_handling2()
    rtt_matrix = rtt_handling2()
    new_avg_pfc = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)

    new_avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
    new_avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
    utility_breakdown = calculate_utility_breakdown(
        new_avg_throughput,
        new_avg_rtt,
        new_avg_pfc,
        throughput_weight,
        rtt_weight,
        pfc_weight,
    )
    utility_value = utility_breakdown['total_utility']
    output_metric(stop_time_this_round, new_avg_throughput, new_avg_rtt, new_avg_pfc, utility_value)
    log_cycle_summary(
        stop_time_this_round,
        flow_ratio_snapshot,
        utility_breakdown,
        current_solution,
        {
            'state': 'collect_only',
            'mode': parameter_mode,
            'trigger_reason': trigger_reason,
            'tuning_round_index': tuning_round_index,
            'tuning_round': 0,
            'candidate_type': 'current',
        },
        previous_flow_ratio_snapshot,
        kl_divergence,
    )

def aggressive_tuning(throughput_weight, rtt_weight, pfc_weight, current_time, flow_ratio_snapshot,
                      previous_flow_ratio_snapshot, kl_divergence, trigger_reason, tuning_round_index):
    start_time_this_round = current_time - t_tune
    stop_time_this_round = current_time

    current_solution = load_current_parameters()
    throughput_matrix = throughput_handling2()
    rtt_matrix = rtt_handling2()
    # pfc_matrix = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)
    new_avg_pfc = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)

    new_avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
    new_avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
    # if np.any(pfc_matrix > 0):
    #     new_avg_pfc = np.mean(pfc_matrix[pfc_matrix > 0])
    # else:
    #     new_avg_pfc = 0

    baseline_utility_breakdown = calculate_utility_breakdown(
        new_avg_throughput,
        new_avg_rtt,
        new_avg_pfc,
        throughput_weight,
        rtt_weight,
        pfc_weight,
    )
    baseline_value = baseline_utility_breakdown['total_utility']
    current_value = baseline_value
    output_metric(stop_time_this_round, new_avg_throughput, new_avg_rtt, new_avg_pfc, current_value)
    log_cycle_summary(
        stop_time_this_round,
        flow_ratio_snapshot,
        baseline_utility_breakdown,
        current_solution,
        {
            'state': 'baseline',
            'mode': 'aggressive',
            'trigger_reason': trigger_reason,
            'tuning_round_index': tuning_round_index,
            'tuning_round': 0,
            'temperature': initial_temperature,
            'candidate_type': 'current',
        },
        previous_flow_ratio_snapshot,
        kl_divergence,
    )

    best_solution = current_solution.copy()
    best_value = baseline_value
    temperature = initial_temperature

    tuning_rounds = 0
    candidate_type = 'default'

    new_solution = generate_new_parameters('default', current_solution)
    write_parameters(new_solution)

    while temperature > final_temperature:
        for i in range(attempt_times):
            start_time_this_round = stop_time_this_round
            stop_time_this_round += t_tune
            wait_for_throughput_snapshot_after(stop_time_this_round)
            
            throughput_matrix = throughput_handling2()
            rtt_matrix = rtt_handling2()
            # pfc_matrix = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)
            new_avg_pfc = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)
            new_avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
            new_avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
            # if np.any(pfc_matrix > 0):
            #     new_avg_pfc = np.mean(pfc_matrix[pfc_matrix > 0])
            # else:
            #     new_avg_pfc = 0

            evaluated_solution = new_solution.copy()
            evaluated_candidate_type = candidate_type
            utility_breakdown = calculate_utility_breakdown(
                new_avg_throughput,
                new_avg_rtt,
                new_avg_pfc,
                throughput_weight,
                rtt_weight,
                pfc_weight,
            )
            new_value = utility_breakdown['total_utility']
            output_metric(stop_time_this_round, new_avg_throughput, new_avg_rtt, new_avg_pfc, new_value)
            
            delta = new_value - current_value
            update_bad_rate_ewma(delta)
            accepted = False

            if delta > 0 or math.exp(delta / temperature) > random.random():
                current_solution = new_solution.copy()
                current_value = new_value
                accepted = True

            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value

            log_cycle_summary(
                stop_time_this_round,
                flow_ratio_snapshot,
                utility_breakdown,
                evaluated_solution,
                {
                    'state': 'annealing_round',
                    'mode': 'aggressive',
                    'tuning_round_index': tuning_round_index,
                    'tuning_round': tuning_rounds + 1,
                    'temperature': temperature,
                    'candidate_type': evaluated_candidate_type,
                    'accepted': accepted,
                    'best_value': best_value,
                },
                previous_flow_ratio_snapshot,
                kl_divergence,
            )

            new_solution = generate_new_parameters('aggressive', current_solution)
            candidate_type = 'aggressive'
            write_parameters(new_solution)
        
            tuning_rounds += 1

        temperature *= cooling_rate
        
    write_parameters(best_solution)


def conservative_tuning(throughput_weight, rtt_weight, pfc_weight, current_time, flow_ratio_snapshot,
                        previous_flow_ratio_snapshot, kl_divergence, trigger_reason, tuning_round_index):
    start_time_this_round = current_time - t_tune
    stop_time_this_round = current_time

    current_solution = load_current_parameters()
    throughput_matrix = throughput_handling2()
    rtt_matrix = rtt_handling2()
    # pfc_matrix = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)
    new_avg_pfc = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)

    new_avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
    new_avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
    # if np.any(pfc_matrix > 0):
    #     new_avg_pfc = np.mean(pfc_matrix[pfc_matrix > 0])
    # else:
    #     new_avg_pfc = 0

    baseline_utility_breakdown = calculate_utility_breakdown(
        new_avg_throughput,
        new_avg_rtt,
        new_avg_pfc,
        throughput_weight,
        rtt_weight,
        pfc_weight,
    )
    baseline_value = baseline_utility_breakdown['total_utility']
    current_value = baseline_value
    output_metric(stop_time_this_round, new_avg_throughput, new_avg_rtt, new_avg_pfc, current_value)
    log_cycle_summary(
        stop_time_this_round,
        flow_ratio_snapshot,
        baseline_utility_breakdown,
        current_solution,
        {
            'state': 'baseline',
            'mode': 'conservative',
            'trigger_reason': trigger_reason,
            'tuning_round_index': tuning_round_index,
            'tuning_round': 0,
            'temperature': initial_temperature,
            'candidate_type': 'current',
        },
        previous_flow_ratio_snapshot,
        kl_divergence,
    )

    best_solution = current_solution.copy()
    best_value = baseline_value
    temperature = initial_temperature

    previous_avg_throughput = new_avg_throughput
    previous_avg_rtt = new_avg_rtt
    tuning_rounds = 0
    candidate_type = 'default'

    new_solution = generate_new_parameters('default', current_solution)
    write_parameters(new_solution)

    while temperature > final_temperature:
        for i in range(attempt_times):
            start_time_this_round = stop_time_this_round
            stop_time_this_round += t_tune
            wait_for_throughput_snapshot_after(stop_time_this_round)
            
            throughput_matrix = throughput_handling2()
            rtt_matrix = rtt_handling2()
            # pfc_matrix = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)
            new_avg_pfc = pfc_handling(start_time_this_round, stop_time_this_round, pfc_start_line)

            new_avg_throughput = mean_or_default(throughput_matrix[throughput_matrix > 5], 0)
            new_avg_rtt = mean_or_default(rtt_matrix[rtt_matrix > 0], base_rtt)
            # if np.any(pfc_matrix > 0):
            #     new_avg_pfc = np.mean(pfc_matrix[pfc_matrix > 0])
            # else:
            #     new_avg_pfc = 0

            evaluated_solution = new_solution.copy()
            evaluated_candidate_type = candidate_type
            utility_breakdown = calculate_utility_breakdown(
                new_avg_throughput,
                new_avg_rtt,
                new_avg_pfc,
                throughput_weight,
                rtt_weight,
                pfc_weight,
            )
            new_value = utility_breakdown['total_utility']
            output_metric(stop_time_this_round, new_avg_throughput, new_avg_rtt, new_avg_pfc, new_value)
            
            delta = new_value - current_value
            update_bad_rate_ewma(delta)
            accepted = False

            if delta > 0 or math.exp(delta / temperature) > random.random():
                current_solution = new_solution.copy()
                current_value = new_value
                accepted = True

            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value

            log_cycle_summary(
                stop_time_this_round,
                flow_ratio_snapshot,
                utility_breakdown,
                evaluated_solution,
                {
                    'state': 'annealing_round',
                    'mode': 'conservative',
                    'tuning_round_index': tuning_round_index,
                    'tuning_round': tuning_rounds + 1,
                    'temperature': temperature,
                    'candidate_type': evaluated_candidate_type,
                    'accepted': accepted,
                    'best_value': best_value,
                },
                previous_flow_ratio_snapshot,
                kl_divergence,
            )
            
            new_solution = generate_new_parameters('conservative', current_solution)
            candidate_type = 'conservative'
            write_parameters(new_solution)
                    
            previous_avg_throughput = new_avg_throughput
            previous_avg_rtt = new_avg_rtt
            tuning_rounds += 1
        
        temperature *= cooling_rate

    write_parameters(best_solution)

def start_tuning(current_time, large_flow_num, small_flow_num, kl_divergence, trigger_reason, trigger_flow_ratio, previous_flow_ratio_snapshot):
    global total_tuning_rounds, is_tuning
    total_tuning_rounds += 1

    parameter_mode, throughput_weight, rtt_weight, pfc_weight = judge_mode(large_flow_num, small_flow_num)

    if trigger_reason == 'initial_observation':
        current_time += t_tune
        wait_for_throughput_snapshot_after(current_time)

    if COLLECT_ONLY_MODE:
        collect_metrics_only(
            parameter_mode,
            throughput_weight,
            rtt_weight,
            pfc_weight,
            current_time,
            trigger_flow_ratio,
            previous_flow_ratio_snapshot,
            kl_divergence,
            trigger_reason,
            total_tuning_rounds,
        )
        is_tuning = False
        return

    if parameter_mode == 'aggressive':
        aggressive_tuning(
            throughput_weight,
            rtt_weight,
            pfc_weight,
            current_time,
            trigger_flow_ratio,
            previous_flow_ratio_snapshot,
            kl_divergence,
            trigger_reason,
            total_tuning_rounds,
        )
    else:
        conservative_tuning(
            throughput_weight,
            rtt_weight,
            pfc_weight,
            current_time,
            trigger_flow_ratio,
            previous_flow_ratio_snapshot,
            kl_divergence,
            trigger_reason,
            total_tuning_rounds,
        )

    is_tuning = False

sketch_heavypart_path = mix_path('switch_sketch_heavypart.tr')
host_flow_report_path = mix_path('host_flow_report.tr')
window_size = 2
monitor_interval_ms = resolve_positive_ms('PARALEON_MONITOR_INTERVAL_MS', 1.0)
parameter_tuning_interval_ms = resolve_positive_ms('PARALEON_PARAMETER_TUNING_INTERVAL_MS', 1.002)
t_reset = monitor_interval_ms / 1000.0
large_flow_threshold = 1024
large_flow_threshold_bytes = large_flow_threshold * 1024
trigger_threshold = 0.01
is_tuning = False
switch_flowid_status_dict = {}
previous_observation_hash = None
previous_flow_ratio = {'large': -1, 'small': -1}
current_flow_ratio = {'large': 0, 'small': 0}
current_flow_ratio_metadata = {}
previous_flow_ratio_metadata = {}
HOST_FLOW_CHUNK_BYTES = resolve_positive_int('PARALEON_PLUS_CHUNK_BYTES', 1024 * 1024)

AGGRESSIVE_WEIGHTS = resolve_mode_weights('PARALEON_AGGRESSIVE_WEIGHTS', (0.6, 0.2, 0.2))
CONSERVATIVE_WEIGHTS = resolve_mode_weights('PARALEON_CONSERVATIVE_WEIGHTS', (0.3, 0.5, 0.2))
AGGRESSIVE_MODE_LARGE_FLOW_THRESHOLD = resolve_ratio_threshold('PARALEON_AGGRESSIVE_MODE_LARGE_FLOW_THRESHOLD', 0.5)
DIRECTION_CAP_POLICY = os.environ.get('PARALEON_DIRECTION_CAP_POLICY', 'adaptive_ewma')
DIRECTION_PROB_CAP_MAX = resolve_unit_interval('PARALEON_DIRECTION_PROB_CAP', 0.7)
DIRECTION_PROB_CAP_MIN = 0.5
if DIRECTION_PROB_CAP_MAX < DIRECTION_PROB_CAP_MIN:
    raise ValueError('PARALEON_DIRECTION_PROB_CAP must be >= 0.5')
if DIRECTION_CAP_POLICY not in ('fixed', 'adaptive_ewma'):
    raise ValueError('PARALEON_DIRECTION_CAP_POLICY must be fixed or adaptive_ewma')
COLLECT_ONLY_MODE = os.environ.get('PARALEON_COLLECT_ONLY') == '1'
STEP_SCALE = resolve_positive_float('PARALEON_TUNER_STEP_SCALE', 1.0)
MUTATE_COUNT = resolve_positive_int('PARALEON_TUNER_MUTATE_COUNT', len(PARAMETER_SPECS))
DIRECTION_BAD_EWMA_ALPHA = resolve_open_unit_interval('PARALEON_DIRECTION_BAD_EWMA_ALPHA', 0.15)
DIRECTION_DELTA_EPS = resolve_positive_float('PARALEON_DIRECTION_DELTA_EPS', 0.01)
bad_rate_ewma = 0.0

base_throughput = 100
base_rtt = 40
t_tune = parameter_tuning_interval_ms / 1000.0
monitor_window_seconds = monitor_interval_ms / 1000.0
# pfc_pause_time = 0.000005
total_directed_links = load_total_directed_links()

throughput_input_file = mix_path('switch_portrate.tr')
rtt_input_file = mix_path('rtt.tr')
pfc_input_file = mix_path('pfc.txt')
metric_output_file = mix_path('metric_output.tr')
monitor_output_file = mix_path('monitor_output.tr')

pfc_start_line = 0
pfc_active_links = set()
observation_pfc_start_line = 0
observation_pfc_active_links = set()
parameter_output_file = mix_path('parameter.txt')


initial_temperature = resolve_positive_float('PARALEON_TUNER_INITIAL_TEMPERATURE', 0.1)
final_temperature = resolve_positive_float('PARALEON_TUNER_FINAL_TEMPERATURE', 0.01)
attempt_times = resolve_positive_int('PARALEON_TUNER_ATTEMPT_TIMES', 8)
cooling_rate = resolve_open_unit_interval('PARALEON_TUNER_COOLING_RATE', 0.76)

if final_temperature > initial_temperature:
    raise ValueError('PARALEON_TUNER_FINAL_TEMPERATURE cannot exceed PARALEON_TUNER_INITIAL_TEMPERATURE')

tuner_seed = os.environ.get('PARALEON_TUNER_SEED')
if tuner_seed is not None and tuner_seed.strip():
    random.seed(int(tuner_seed))
    np.random.seed(int(tuner_seed) % (2 ** 32))

total_tuning_rounds = 0
total_monitor_rounds = 0

if not os.path.exists(parameter_output_file) or os.path.getsize(parameter_output_file) == 0:
    write_parameters(default_parameter_values())

if os.path.exists(metric_output_file):
    os.remove(metric_output_file)
if os.path.exists(monitor_output_file):
    os.remove(monitor_output_file)
 

while True:
    observation_path = host_flow_report_path if USE_HOST_FLOW_REPORT else sketch_heavypart_path
    current_observation_hash = calculate_file_hash(observation_path)
    if current_observation_hash != previous_observation_hash:
        previous_observation_hash = current_observation_hash
        total_monitor_rounds += 1
        current_flow_ratio_metadata = {}

        if USE_HOST_FLOW_REPORT:
            current_time, current_observation = load_host_flow_report()
            if current_time is None or not current_observation:
                time.sleep(1)
                continue
            current_ratio, current_flow_ratio_metadata = classify_host_flow_progress(
                current_observation,
            )
            current_flow_ratio.update(current_ratio)
        else:
            current_time, current_flowid_size_dict = load_sketch()
            if current_time is None or not any(flowid_size_dict for flowid_size_dict in current_flowid_size_dict.values()):
                time.sleep(1)
                continue

            update_all_flows(current_flowid_size_dict, current_time)
            switch_upload_dict = filter_flows()
            current_flow_ratio_metadata = upload_ratio(switch_upload_dict, current_time)

        kl_divergence = compute_divergence(current_flow_ratio, previous_flow_ratio)
        current_flow_ratio_snapshot = dict(current_flow_ratio)
        current_flow_ratio_snapshot.update(current_flow_ratio_metadata)
        previous_flow_ratio_snapshot = dict(previous_flow_ratio)
        previous_flow_ratio_snapshot.update(previous_flow_ratio_metadata)
        trigger_reason = None
        if COLLECT_ONLY_MODE:
            if is_tuning == False:
                if previous_flow_ratio['small'] == -1:
                    trigger_reason = 'initial_observation'
                else:
                    trigger_reason = 'observation'
        else:
            if previous_flow_ratio['small'] == -1 and is_tuning == False:
                trigger_reason = 'initial_observation'
            elif kl_divergence > trigger_threshold and is_tuning == False:
                trigger_reason = 'kl_divergence'

        collect_monitor_cycle_summary(
            current_time,
            current_flow_ratio_snapshot,
            current_flow_ratio_metadata,
            previous_flow_ratio_snapshot,
            kl_divergence,
            trigger_reason,
        )

        if trigger_reason is not None:
            is_tuning = True
            tuning_thread = threading.Thread(
                target=start_tuning,
                args=(
                    current_time,
                    current_flow_ratio_snapshot['large'],
                    current_flow_ratio_snapshot['small'],
                    kl_divergence,
                    trigger_reason,
                    current_flow_ratio_snapshot,
                    previous_flow_ratio_snapshot,
                ),
            )
            tuning_thread.start()
            
        previous_flow_ratio.update(current_flow_ratio)
        previous_flow_ratio_metadata = dict(current_flow_ratio_metadata)
        current_flow_ratio['large'] = 0
        current_flow_ratio['small'] = 0

    else:
        time.sleep(3)
