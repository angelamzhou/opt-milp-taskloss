import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reweighted_mse_helpers import (
    ShortestPathOracle,
    convert_grid_to_list,
    generate_data,
    generateInstanceDict,
    run_replication_over_weights,
)


RESULT_COLUMNS = [
    'task_id',
    'preset',
    'replication',
    'n_train',
    'polykernel_degree',
    'n_test',
    'time',
    'algo',
    'tr_regret',
    'tst_regret',
    'reweight',
    'mixture_weight',
    'avg_raw_weight',
    'avg_fitted_weight',
]


DEFAULTS = {
    'n_train_values': [100, 1000, 2000],
    'degrees': [1, 2, 4, 6, 8],
    'mixture_weights': [0.2, 0.4, 0.6, 0.8],
    'n_reps': 8,
    'n_test': 10000,
    'n_holdout': 500,
    'num_reweights': 3,
    'skip_spo': False,
}


PRESETS = {
    'custom': {},
    'current': {
        **DEFAULTS,
    },
    'may19': {
        'n_train_values': [100, 1000, 2000],
        'degrees': [1, 2, 4, 6, 8],
        'mixture_weights': [0.2, 0.4, 0.6, 0.8],
        'n_reps': 8,
        'n_test': 10000,
        'n_holdout': 500,
        'num_reweights': 3,
        'skip_spo': False,
    },
    'deprecated-paper': {
        'n_train_values': [100, 250, 500, 1000, 1500, 2000],
        'degrees': [1, 2, 4, 6, 8],
        'mixture_weights': [0.2, 0.3975, 0.595, 0.7925, 0.99],
        'n_reps': 12,
        'n_test': 10000,
        'n_holdout': 1000,
        'num_reweights': 1,
        'skip_spo': True,
    },
    'paper-figure5': {
        'n_train_values': [100, 1000, 2000],
        'degrees': [1, 2, 4, 6, 8],
        'mixture_weights': [0.2, 0.3975, 0.595, 0.7925, 0.99],
        'n_reps': 50,
        'n_test': 10000,
        'n_holdout': 1000,
        'num_reweights': 1,
        'skip_spo': False,
    },
}


def parse_int_list(value):
    return [int(part) for part in value.split(',') if part]


def parse_float_list(value):
    return [float(part) for part in value.split(',') if part]


def get_weight_model(weight_model):
    if weight_model == 'ridgecv':
        return None, None
    if weight_model == 'linear':
        return LinearRegression, None
    if weight_model == 'rf':
        return RandomForestRegressor(random_state=1), {
            'n_estimators': [100, 200],
            'max_depth': [3, None],
            'min_samples_leaf': [1, 5]
        }
    raise ValueError('unknown weight model: %s' % weight_model)


def build_graph_params(grid_dim):
    [sources, destinations, scalar_nodes, tuple_nodes] = convert_grid_to_list(grid_dim, grid_dim)
    nodes = np.unique(list(set(sources).union(set(destinations))))
    return {
        'nodes': nodes,
        'sources': sources,
        'destinations': destinations,
        'start_node': scalar_nodes[(0, 0)],
        'end_node': scalar_nodes[(grid_dim - 1, grid_dim - 1)]
    }


def apply_preset(args):
    preset = DEFAULTS if args.preset == 'custom' else PRESETS[args.preset]
    if args.n_train_values is None:
        args.n_train_values = preset['n_train_values']
    if args.degrees is None:
        args.degrees = preset['degrees']
    if args.mixture_weights is None:
        args.mixture_weights = preset['mixture_weights']
    if args.context_mixture_weights is None:
        args.context_mixture_weights = list(args.mixture_weights)
    if args.n_reps is None:
        args.n_reps = preset['n_reps']
    if args.n_test is None:
        args.n_test = preset['n_test']
    if args.n_holdout is None:
        args.n_holdout = preset['n_holdout']
    if args.num_reweights is None:
        args.num_reweights = preset['num_reweights']
    if args.use_preset_skip_spo:
        args.skip_spo = preset['skip_spo']
    return args


def expected_rows_per_task(args):
    rows = 1
    rows += len(args.mixture_weights) * args.num_reweights
    rows += len(args.context_mixture_weights or [])
    rows += 0 if args.skip_spo else 1
    return rows


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return '%dh%02dm%02ds' % (hours, minutes, secs)
    if minutes:
        return '%dm%02ds' % (minutes, secs)
    return '%ds' % secs


def task_id_for(degree, n_train, replication):
    return 'deg%s_n%s_rep%s' % (degree, n_train, replication)


def annotate_rows(rows, task_id, preset, replication):
    annotated = []
    for row in rows:
        row_copy = dict(row)
        row_copy['task_id'] = task_id
        row_copy['preset'] = preset
        row_copy['replication'] = replication
        annotated.append(row_copy)
    return annotated


def append_rows(output_path, rows):
    frame = pd.DataFrame.from_records(rows)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[RESULT_COLUMNS]
    header = not output_path.exists()
    frame.to_csv(output_path, mode='a', header=header, index=False)


def clean_and_resume_output(output_path, expected_row_count):
    if not output_path.exists():
        return set()

    existing = pd.read_csv(output_path)
    if existing.empty:
        output_path.unlink()
        return set()
    if 'task_id' not in existing.columns:
        raise ValueError('existing output file is not resumable because it lacks a task_id column')

    counts = existing.groupby('task_id').size()
    complete_ids = set(counts[counts == expected_row_count].index.tolist())
    keep_mask = existing['task_id'].isin(complete_ids)
    if not keep_mask.all():
        cleaned = existing.loc[keep_mask].copy()
        if cleaned.empty:
            output_path.unlink()
        else:
            cleaned.to_csv(output_path, index=False)
    return complete_ids


def build_tasks(args):
    tasks = []
    for degree in args.degrees:
        for n_train in args.n_train_values:
            for replication in range(args.n_reps):
                tasks.append({
                    'degree': degree,
                    'n_train': n_train,
                    'replication': replication,
                    'task_id': task_id_for(degree, n_train, replication),
                })
    return tasks


def run_single_task(task, args, graph_params, b_true, X_test_cache, c_test_cache,
                    test_dict_cache, regressor, weight_regressor, weight_param_grid):
    degree = task['degree']
    n_train = task['n_train']
    data_params = [
        n_train,
        args.n_test,
        args.n_holdout,
        degree,
        args.polykernel_noise_half_width,
        b_true
    ]
    return run_replication_over_weights(
        data_params,
        X_test_cache[degree],
        c_test_cache[degree],
        test_dict_cache[degree],
        args.mixture_weights,
        regressor,
        graph_params,
        num_reweights=args.num_reweights,
        random_regr=False,
        context_mixture_weights=args.context_mixture_weights,
        weight_regressor=weight_regressor,
        weight_param_grid=weight_param_grid,
        context_n_folds=args.context_folds,
        context_weight_cv=args.context_weight_cv,
        context_cap_quantile=args.context_cap_quantile,
        context_max_weight=args.context_max_weight,
        run_spo=not args.skip_spo
    )


def run_experiment(args):
    np.random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    graph_params = build_graph_params(args.grid_dim)
    oracle = ShortestPathOracle(graph_params)
    d_feasibleregion = len(graph_params['sources'])
    b_true = np.random.binomial(1, 0.5, size=(d_feasibleregion, args.p_features))

    regressor = LinearRegression
    weight_regressor, weight_param_grid = get_weight_model(args.weight_model)

    X_test_cache = {}
    c_test_cache = {}
    test_dict_cache = {}
    for degree in args.degrees:
        [X_train, c_train, X_validation, c_validation, X_test, c_test] = generate_data(
            max(args.n_train_values),
            args.n_test,
            args.n_holdout,
            degree,
            args.polykernel_noise_half_width,
            b_true,
            gen_test=True
        )
        X_test_cache[degree] = X_test
        c_test_cache[degree] = c_test
        test_dict_cache[degree] = generateInstanceDict(X_test, c_test, oracle)

    all_tasks = build_tasks(args)
    expected_row_count = expected_rows_per_task(args)
    completed_task_ids = clean_and_resume_output(output_path, expected_row_count)
    pending_tasks = [task for task in all_tasks if task['task_id'] not in completed_task_ids]

    total_tasks = len(all_tasks)
    completed_tasks = len(completed_task_ids)
    print(
        'starting %d total tasks, %d already complete, %d remaining' % (
            total_tasks, completed_tasks, len(pending_tasks)
        ),
        flush=True,
    )

    if not pending_tasks:
        return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)

    if args.batch_size is None:
        if args.n_jobs == -1:
            batch_size = max(1, math.ceil(len(pending_tasks) / max(1, min(8, len(pending_tasks)))))
        else:
            batch_size = max(1, args.n_jobs)
    else:
        batch_size = max(1, args.batch_size)

    start_time = time.time()
    for batch_start in range(0, len(pending_tasks), batch_size):
        batch = pending_tasks[batch_start:batch_start + batch_size]
        batch_results = Parallel(
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            prefer=args.parallel_backend,
        )(
            delayed(run_single_task)(
                task,
                args,
                graph_params,
                b_true,
                X_test_cache,
                c_test_cache,
                test_dict_cache,
                regressor,
                weight_regressor,
                weight_param_grid,
            )
            for task in batch
        )

        for task, rows in zip(batch, batch_results):
            annotated_rows = annotate_rows(rows, task['task_id'], args.preset, task['replication'])
            append_rows(output_path, annotated_rows)
            completed_tasks += 1

            elapsed = time.time() - start_time
            rate = elapsed / max(1, completed_tasks - len(completed_task_ids))
            remaining = total_tasks - completed_tasks
            eta_seconds = rate * remaining
            pct = 100.0 * completed_tasks / total_tasks
            print(
                '[%6.2f%%] completed %d/%d tasks | elapsed %s | eta %s | last task %s' % (
                    pct,
                    completed_tasks,
                    total_tasks,
                    format_seconds(elapsed),
                    format_seconds(eta_seconds),
                    task['task_id'],
                ),
                flush=True,
            )

    return pd.read_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), default='custom')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--grid-dim', type=int, default=5)
    parser.add_argument('--p-features', type=int, default=5)
    parser.add_argument('--n-train-values', type=parse_int_list, default=None)
    parser.add_argument('--degrees', type=parse_int_list, default=None)
    parser.add_argument('--mixture-weights', type=parse_float_list, default=None)
    parser.add_argument('--context-mixture-weights', type=parse_float_list, default=None)
    parser.add_argument('--n-reps', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--parallel-backend', choices=['threads', 'processes'], default='processes')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--num-reweights', type=int, default=None)
    parser.add_argument('--n-test', type=int, default=None)
    parser.add_argument('--n-holdout', type=int, default=None)
    parser.add_argument('--polykernel-noise-half-width', type=float, default=0.5)
    parser.add_argument('--weight-model', choices=['ridgecv', 'linear', 'rf'], default='ridgecv')
    parser.add_argument('--context-folds', type=int, default=5)
    parser.add_argument('--context-weight-cv', type=int, default=3)
    parser.add_argument('--context-cap-quantile', type=float, default=0.9)
    parser.add_argument('--context-max-weight', type=float, default=1.0)
    parser.add_argument('--use-preset-skip-spo', action='store_true')
    parser.add_argument('--skip-spo', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--output', default='results/context_reweighting.csv')
    args = parser.parse_args()
    args = apply_preset(args)

    results_df = run_experiment(args)
    print('wrote', args.output)
    print(results_df.groupby('algo')['tst_regret'].mean())


if __name__ == '__main__':
    main()
