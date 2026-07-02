#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use('Agg')

import matplotlib.pyplot as plt


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


def parse_experiment_id(raw_value):
    return int(str(raw_value).strip())


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


def resolve_tuner_log_path(args):
    if args.log:
        log_path = Path(args.log).expanduser().resolve()
        if not log_path.exists():
            raise SystemExit("tuner log does not exist: %s" % log_path)
        return log_path

    if args.id is None:
        raise SystemExit("please provide either --id or --log")

    run_dir = resolve_run_dir(parse_experiment_id(args.id))
    manifest = load_manifest(run_dir)
    tuner_log = manifest.get("files", {}).get("tuner_log") or manifest.get("files", {}).get("controller_log")
    if tuner_log:
        log_path = Path(tuner_log)
        if log_path.exists():
            return log_path

    fallback = run_dir / "tuner.log"
    if fallback.exists():
        return fallback
    raise SystemExit("no tuner log found for experiment %s" % args.id)


def split_top_level_commas(text):
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth = max(0, depth - 1)
        elif char == ',' and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def parse_key_value_line(text):
    values = {}
    for part in split_top_level_commas(text):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        values[key.strip()] = parse_scalar(value)
    return values


UTILITY_SEGMENT_RE = re.compile(
    r'^(?P<name>\w+)=(?P<norm>[^\s]+) \(raw=(?P<raw>[^,]+), weighted=(?P<weighted>[^)]+)\)$'
)


def parse_utility_line(text):
    values = {}
    for part in split_top_level_commas(text):
        match = UTILITY_SEGMENT_RE.match(part)
        if match:
            name = match.group('name')
            values[name] = parse_scalar(match.group('norm'))
            prefix = name.replace('_norm', '') if name.endswith('_norm') else name
            values[prefix + '_raw'] = parse_scalar(match.group('raw'))
            values[prefix + '_weighted'] = parse_scalar(match.group('weighted'))
            continue
        if '=' in part:
            key, value = part.split('=', 1)
            values[key.strip()] = parse_scalar(value)
    return values


def parse_block(block_text):
    lines = [line.rstrip() for line in block_text.splitlines() if line.strip()]
    if not lines:
        return None
    if '=' not in lines[0]:
        return None

    first_key, first_value = lines[0].split('=', 1)
    block = {first_key.strip(): parse_scalar(first_value)}

    for line in lines[1:]:
        stripped = line.strip()
        if ':' not in stripped:
            continue
        prefix, content = stripped.split(':', 1)
        content = content.strip()
        if prefix == 'utility':
            block['utility'] = parse_utility_line(content)
        else:
            block[prefix] = parse_key_value_line(content)
    return block


def load_tuning_blocks(log_path):
    text = log_path.read_text(encoding='utf-8')
    blocks = []
    for raw_block in text.split('\n\n'):
        block = parse_block(raw_block)
        if block is not None:
            blocks.append(block)
    if not blocks:
        raise SystemExit("no tuning blocks found in %s" % log_path)
    return blocks


def filter_blocks_by_time(blocks, start_time=None, end_time=None):
    filtered_blocks = []
    for block in blocks:
        sim_time = block.get('sim_time')
        if sim_time is None:
            continue
        sim_time = float(sim_time)
        if start_time is not None and sim_time < start_time:
            continue
        if end_time is not None and sim_time > end_time:
            continue
        filtered_blocks.append(block)
    if not filtered_blocks:
        raise SystemExit('no tuning blocks remain after applying time filter')
    return filtered_blocks


def format_time_node(sim_time):
    return ('%.6f' % float(sim_time)).rstrip('0').rstrip('.')


def build_parameter_dataframe(blocks):
    parameter_names = []
    seen = set()
    for block in blocks:
        parameters = dict(block.get('parameters', {}))
        for parameter_name in parameters:
            if parameter_name in seen:
                continue
            seen.add(parameter_name)
            parameter_names.append(parameter_name)

    values_by_time = {}
    for block in blocks:
        values_by_time[format_time_node(block.get('sim_time'))] = dict(block.get('parameters', {}))

    parameter_df = pd.DataFrame(values_by_time)
    if parameter_names:
        parameter_df = parameter_df.reindex(parameter_names)
    parameter_df.index.name = 'parameter'
    return parameter_df


def build_utility_dataframe(blocks):
    rows = []
    previous_total = None
    for block in blocks:
        utility = dict(block.get('utility', {}))
        tuning = block.get('tuning', {})
        row = {
            'sim_time': block.get('sim_time'),
            'session': tuning.get('session'),
            'round': tuning.get('round'),
            'state': tuning.get('state'),
            'mode': tuning.get('mode'),
            'candidate': tuning.get('candidate'),
            'accepted': tuning.get('accepted'),
            'best_utility': tuning.get('best_utility'),
        }
        row.update(utility)
        total_value = utility.get('total')
        row['delta_total'] = None if previous_total is None or total_value is None else total_value - previous_total
        rows.append(row)
        previous_total = total_value
    return pd.DataFrame(rows)


def build_parameter_plot_dataframe(blocks):
    rows = []
    parameter_names = []
    seen = set()
    for block in blocks:
        utility = dict(block.get('utility', {}))
        parameters = dict(block.get('parameters', {}))
        row = {'sim_time': block.get('sim_time')}
        row.update(utility)
        row.update(parameters)
        rows.append(row)
        for parameter_name in parameters:
            if parameter_name in seen:
                continue
            seen.add(parameter_name)
            parameter_names.append(parameter_name)
    return pd.DataFrame(rows), parameter_names


def build_output_stem(args, log_path):
    if args.id is not None:
        source_name = 'exp_%s' % parse_experiment_id(args.id)
    else:
        source_name = '%s_%s' % (log_path.parent.name, log_path.stem)

    suffix_parts = []
    if args.start_time is not None:
        suffix_parts.append('from_%s' % format_time_node(args.start_time).replace('.', 'p'))
    if args.end_time is not None:
        suffix_parts.append('to_%s' % format_time_node(args.end_time).replace('.', 'p'))
    if suffix_parts:
        source_name = '%s_%s' % (source_name, '_'.join(suffix_parts))

    return re.sub(r'[^A-Za-z0-9._-]+', '_', source_name)


def build_time_window_label(args):
    if args.start_time is None and args.end_time is None:
        return None

    if args.start_time is not None and args.end_time is not None:
        return '%.6g <= t <= %.6g s' % (args.start_time, args.end_time)
    if args.start_time is not None:
        return 't >= %.6g s' % args.start_time
    return 't <= %.6g s' % args.end_time


def build_plot_output_path(args, log_path):
    return SCRIPT_DIR / ('utility_history_%s.png' % build_output_stem(args, log_path))


def build_parameter_scatter_dir(args, log_path):
    return SCRIPT_DIR / ('parameter_scatter_%s' % build_output_stem(args, log_path))


def build_plot_title(args, log_path):
    if args.id is not None:
        title = 'Utility History (Experiment %s)' % parse_experiment_id(args.id)
    else:
        title = 'Utility History (%s)' % log_path.parent.name

    time_window_label = build_time_window_label(args)
    if time_window_label:
        title = '%s, %s' % (title, time_window_label)
    return title


def plot_utility_history(utility_df, output_path, title):
    plot_columns = [
        ('throughput_norm', 'throughput_norm'),
        ('rtt_norm', 'rtt_norm'),
        ('pfc_norm', 'pfc_norm'),
        ('total', 'total'),
    ]
    missing_columns = [column for column, _ in plot_columns if column not in utility_df.columns]
    if missing_columns:
        raise SystemExit('missing utility columns for plotting: %s' % ', '.join(missing_columns))

    plot_df = utility_df[['sim_time'] + [column for column, _ in plot_columns]].copy()
    plot_df['sim_time'] = pd.to_numeric(plot_df['sim_time'], errors='coerce')
    for column, _ in plot_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors='coerce')
    plot_df = plot_df.dropna(subset=['sim_time'])
    if plot_df.empty:
        raise SystemExit('no utility samples available for plotting')

    figure, axis = plt.subplots(figsize=(10, 5))
    for column, label in plot_columns:
        series = plot_df[['sim_time', column]].dropna()
        if series.empty:
            continue
        axis.plot(series['sim_time'], series[column], linewidth=1.6, label=label)

    if not axis.lines:
        plt.close(figure)
        raise SystemExit('no utility values available for plotting')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis.set_title(title)
    axis.set_xlabel('Simulation Time (s)')
    axis.set_ylabel('Utility')
    axis.grid(True, linestyle='--', alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_parameter_relationships(plot_df, parameter_names, output_dir, title_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_plot in output_dir.glob('*.png'):
        existing_plot.unlink()

    relationship_specs = [
        ('throughput_norm', 'throughput_norm'),
        ('rtt_norm', 'rtt_norm'),
        ('pfc_norm', 'pfc_norm'),
    ]

    plot_df = plot_df.copy()
    numeric_columns = ['sim_time'] + parameter_names + [column for column, _ in relationship_specs]
    for column in numeric_columns:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors='coerce')

    for parameter_name in parameter_names:
        time_series = plot_df[['sim_time', parameter_name]].dropna()
        if time_series.empty:
            continue

        figure, axis = plt.subplots(figsize=(8.5, 4.5))
        axis.plot(time_series['sim_time'], time_series[parameter_name], marker='o', markersize=3.5, linewidth=1.2)
        axis.set_title('%s - %s vs time' % (title_prefix, parameter_name))
        axis.set_xlabel('Simulation Time (s)')
        axis.set_ylabel(parameter_name)
        axis.grid(True, linestyle='--', alpha=0.35)
        figure.tight_layout()
        figure.savefig(output_dir / ('%s_vs_time.png' % parameter_name), dpi=150)
        plt.close(figure)

        for utility_column, utility_label in relationship_specs:
            relation_df = plot_df[[parameter_name, utility_column]].dropna()
            if relation_df.empty:
                continue

            figure, axis = plt.subplots(figsize=(6.5, 5))
            axis.scatter(relation_df[parameter_name], relation_df[utility_column], s=28, alpha=0.85)
            axis.set_title('%s - %s vs %s' % (title_prefix, parameter_name, utility_label))
            axis.set_xlabel(parameter_name)
            axis.set_ylabel(utility_label)
            axis.grid(True, linestyle='--', alpha=0.35)
            figure.tight_layout()
            figure.savefig(output_dir / ('%s_vs_%s.png' % (parameter_name, utility_column)), dpi=150)
            plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description='Show Paraleon tuning parameter evolution and utility history from tuner.log')
    parser.add_argument('--id', help='experiment id under mix/')
    parser.add_argument('--log', help='explicit path to tuner.log')
    parser.add_argument('--start-time', type=float, help='optional lower bound for sim_time filtering')
    parser.add_argument('--end-time', type=float, help='optional upper bound for sim_time filtering')
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = resolve_tuner_log_path(args)
    blocks = load_tuning_blocks(log_path)
    blocks = filter_blocks_by_time(blocks, args.start_time, args.end_time)

    parameter_df = build_parameter_dataframe(blocks)
    utility_df = build_utility_dataframe(blocks)
    parameter_plot_df, parameter_names = build_parameter_plot_dataframe(blocks)
    parameter_output_path = SCRIPT_DIR / 'parameter.tsv'
    utility_plot_path = build_plot_output_path(args, log_path)
    parameter_scatter_dir = build_parameter_scatter_dir(args, log_path)
    parameter_df.to_csv(parameter_output_path, sep='\t', float_format='%.2f', na_rep='NA')
    plot_utility_history(utility_df, utility_plot_path, build_plot_title(args, log_path))
    plot_parameter_relationships(parameter_plot_df, parameter_names, parameter_scatter_dir, 'Parameter Plot')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', None)

    print('tuner_log = %s' % log_path)
    print('parameter_file = %s' % parameter_output_path)
    print('utility_plot = %s' % utility_plot_path)
    print('parameter_scatter_dir = %s' % parameter_scatter_dir)
    print()
    print('Utility History')
    print(utility_df.to_string(index=False))


if __name__ == '__main__':
    main()