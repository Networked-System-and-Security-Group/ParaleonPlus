#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use('Agg')

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MIX_DIR = ROOT_DIR / 'mix'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'output'
AVERAGE_THROUGHPUT_CUTOFF = 2.2


def parse_experiment_ids(raw_tokens):
    experiment_ids = []
    seen = set()
    for raw_token in raw_tokens:
        for part in str(raw_token).split(','):
            token = part.strip()
            if not token:
                continue
            experiment_id = int(token)
            if experiment_id in seen:
                continue
            seen.add(experiment_id)
            experiment_ids.append(experiment_id)
    if not experiment_ids:
        raise SystemExit('no experiment ids were provided')
    return experiment_ids


def resolve_run_dirs(experiment_ids):
    indexed_dirs = {}
    for path in MIX_DIR.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split('-', 1)[0]
        if prefix.isdigit():
            indexed_dirs[int(prefix)] = path

    missing_ids = [str(experiment_id) for experiment_id in experiment_ids if experiment_id not in indexed_dirs]
    if missing_ids:
        raise SystemExit('missing experiment ids: %s' % ', '.join(missing_ids))
    return [indexed_dirs[experiment_id] for experiment_id in experiment_ids]


def load_manifest(run_dir):
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        return {
            'id': int(run_dir.name.split('-', 1)[0]),
            'message': run_dir.name,
            'files': {},
            'config': {},
        }
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def collapse_experiment_ids(experiment_ids):
    if not experiment_ids:
        return 'exp_none'

    ranges = []
    start = experiment_ids[0]
    end = experiment_ids[0]
    for experiment_id in experiment_ids[1:]:
        if experiment_id == end + 1:
            end = experiment_id
            continue
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append('%s_%s' % (start, end))
        start = experiment_id
        end = experiment_id

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append('%s_%s' % (start, end))
    return 'exp_' + '_'.join(ranges)


def load_throughput_norm_series(run_dir, drop_first_sample=True, start_time=None, end_time=None):
    csv_path = run_dir / 'throughput_norm.csv'
    if not csv_path.exists():
        raise SystemExit('missing throughput_norm.csv: %s' % csv_path)

    data_frame = pd.read_csv(csv_path)
    if 'time' not in data_frame.columns or 'throughput_norm' not in data_frame.columns:
        raise SystemExit('invalid throughput_norm.csv columns: %s' % csv_path)

    data_frame['time'] = pd.to_numeric(data_frame['time'], errors='coerce')
    data_frame['throughput_norm'] = pd.to_numeric(data_frame['throughput_norm'], errors='coerce')
    data_frame = data_frame.dropna(subset=['time', 'throughput_norm']).sort_values('time').reset_index(drop=True)
    if data_frame.empty:
        raise SystemExit('no numeric throughput_norm rows found: %s' % csv_path)

    if drop_first_sample and len(data_frame.index) > 1:
        data_frame = data_frame.iloc[1:].copy()

    if start_time is not None:
        data_frame = data_frame[data_frame['time'] >= start_time].copy()
    if end_time is not None:
        data_frame = data_frame[data_frame['time'] <= end_time].copy()
    if data_frame.empty:
        raise SystemExit('no throughput_norm rows remain after filtering: %s' % csv_path)

    gap_mask = data_frame['time'].diff().fillna(0) > 0.01
    data_frame.loc[gap_mask, 'throughput_norm'] = float('nan')
    return data_frame


def build_label(experiment_id, manifest, include_message):
    if not include_message:
        return 'exp %s' % experiment_id

    message = manifest.get('message')
    if not message:
        return 'exp %s' % experiment_id
    return 'exp %s (%s)' % (experiment_id, message)


def build_output_stem(experiment_ids, start_time=None, end_time=None, drop_first_sample=True):
    stem = 'throughput_norm_%s' % collapse_experiment_ids(experiment_ids)
    if drop_first_sample:
        stem += '_drop_first'
    if start_time is not None:
        stem += '_from_%s' % ('%.2f' % start_time).rstrip('0').rstrip('.').replace('.', 'p')
    if end_time is not None:
        stem += '_to_%s' % ('%.2f' % end_time).rstrip('0').rstrip('.').replace('.', 'p')
    return stem


def plot_series(series_records, output_path, title, y_max=None, legend_columns=None):
    figure, axis = plt.subplots(figsize=(12, 7))
    for _, label, data_frame in series_records:
        axis.plot(data_frame['time'], data_frame['throughput_norm'], linewidth=1.5, label=label)

    axis.set_title(title)
    axis.set_xlabel('Simulation Time (s)')
    axis.set_ylabel('throughput_norm')
    if y_max is not None:
        axis.set_ylim(0, y_max)
    axis.grid(True, linestyle='--', alpha=0.35)
    if legend_columns is None:
        legend_columns = 3 if len(series_records) > 10 else 2
    axis.legend(ncol=max(1, legend_columns), fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_summary_rows(series_records, average_cutoff=AVERAGE_THROUGHPUT_CUTOFF):
    rows = []
    for experiment_id, _, data_frame in series_records:
        average_frame = data_frame[data_frame['time'] <= average_cutoff]
        values = average_frame['throughput_norm'].dropna()
        if values.empty:
            raise SystemExit(
                'no throughput_norm rows remain for average calculation at or before %.2fs: exp %s'
                % (average_cutoff, experiment_id)
            )
        rows.append(
            {
                'exp_id': experiment_id,
                'end_time': float(data_frame['time'].max()),
                'avg_throughput_norm': float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot throughput_norm.csv for one or more experiments. By default the first sample of each experiment is dropped.'
    )
    parser.add_argument('--ids', nargs='+', required=True, help='experiment ids, e.g. --ids 62 63 64 or --ids 62,63,64')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='directory for the output PNG and TSV files')
    parser.add_argument('--output-name', default=None, help='optional PNG filename; defaults to a name derived from experiment ids')
    parser.add_argument('--summary-name', default=None, help='optional TSV filename; defaults to a name derived from experiment ids')
    parser.add_argument('--title', default=None, help='optional plot title')
    parser.add_argument('--start-time', type=float, default=None, help='optional lower bound for time filtering')
    parser.add_argument('--end-time', type=float, default=None, help='optional upper bound for time filtering')
    parser.add_argument('--keep-first-sample', action='store_true', help='keep the first sample instead of dropping it')
    parser.add_argument('--y-max', type=float, default=None, help='optional y-axis upper bound')
    parser.add_argument('--legend-columns', type=int, default=None, help='optional legend column count')
    parser.add_argument('--no-message-label', action='store_true', help='label lines with only experiment ids')
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_ids = parse_experiment_ids(args.ids)
    run_dirs = resolve_run_dirs(experiment_ids)
    drop_first_sample = not args.keep_first_sample

    series_records = []
    for experiment_id, run_dir in zip(experiment_ids, run_dirs):
        manifest = load_manifest(run_dir)
        data_frame = load_throughput_norm_series(
            run_dir,
            drop_first_sample=drop_first_sample,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        series_records.append(
            (
                experiment_id,
                build_label(experiment_id, manifest, include_message=not args.no_message_label),
                data_frame,
            )
        )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = build_output_stem(
        experiment_ids,
        start_time=args.start_time,
        end_time=args.end_time,
        drop_first_sample=drop_first_sample,
    )
    plot_path = output_dir / (args.output_name or ('%s.png' % output_stem))
    summary_path = output_dir / (args.summary_name or ('%s.tsv' % output_stem))

    title = args.title or 'Throughput Norm over Time (%s)' % ', '.join(str(experiment_id) for experiment_id in experiment_ids)
    if drop_first_sample:
        title += ', first sample removed'

    plot_series(series_records, plot_path, title, y_max=args.y_max, legend_columns=args.legend_columns)
    summary_frame = build_summary_rows(series_records).sort_values('exp_id')
    summary_frame.to_csv(summary_path, sep='\t', index=False, float_format='%.2f')

    print('plot_path = %s' % plot_path)
    print('summary_path = %s' % summary_path)
    print()
    print(summary_frame.to_string(index=False, float_format=lambda value: '%.2f' % value))


if __name__ == '__main__':
    main()
