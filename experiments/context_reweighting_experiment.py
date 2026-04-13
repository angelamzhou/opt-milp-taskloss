import argparse
from pathlib import Path
import sys

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


def run_experiment(args):
    np.random.seed(args.seed)

    graph_params = build_graph_params(args.grid_dim)
    oracle = ShortestPathOracle(graph_params)
    d_feasibleregion = len(graph_params['sources'])
    b_true = np.random.binomial(1, 0.5, size=(d_feasibleregion, args.p_features))

    regressor = LinearRegression
    weight_regressor, weight_param_grid = get_weight_model(args.weight_model)

    results = []
    context_mixture_weights = args.context_mixture_weights or args.mixture_weights

    for degree in args.degrees:
        [X_train, c_train, X_validation, c_validation, X_test, c_test] = generate_data(
            max(args.n_train_values), args.n_test, args.n_holdout, degree,
            args.polykernel_noise_half_width, b_true, gen_test=True
        )
        testDict = generateInstanceDict(X_test, c_test, oracle)

        for n_train in args.n_train_values:
            data_params = [
                n_train,
                args.n_test,
                args.n_holdout,
                degree,
                args.polykernel_noise_half_width,
                b_true
            ]
            replications = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
                delayed(run_replication_over_weights)(
                    data_params,
                    X_test,
                    c_test,
                    testDict,
                    args.mixture_weights,
                    regressor,
                    graph_params,
                    num_reweights=args.num_reweights,
                    random_regr=False,
                    context_mixture_weights=context_mixture_weights,
                    weight_regressor=weight_regressor,
                    weight_param_grid=weight_param_grid,
                    context_n_folds=args.context_folds,
                    context_weight_cv=args.context_weight_cv,
                    context_cap_quantile=args.context_cap_quantile,
                    context_max_weight=args.context_max_weight,
                    run_spo=not args.skip_spo
                )
                for _ in range(args.n_reps)
            )
            results.extend(np.array(replications, dtype=object).flatten())

    return pd.DataFrame.from_records(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--grid-dim', type=int, default=5)
    parser.add_argument('--p-features', type=int, default=5)
    parser.add_argument('--n-train-values', type=parse_int_list, default=parse_int_list('100,1000,2000'))
    parser.add_argument('--degrees', type=parse_int_list, default=parse_int_list('1,2,4,6,8'))
    parser.add_argument('--mixture-weights', type=parse_float_list, default=parse_float_list('0.2,0.4,0.6,0.8'))
    parser.add_argument('--context-mixture-weights', type=parse_float_list, default=None)
    parser.add_argument('--n-reps', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--verbose', type=int, default=20)
    parser.add_argument('--num-reweights', type=int, default=3)
    parser.add_argument('--n-test', type=int, default=10000)
    parser.add_argument('--n-holdout', type=int, default=500)
    parser.add_argument('--polykernel-noise-half-width', type=float, default=0.5)
    parser.add_argument('--weight-model', choices=['ridgecv', 'linear', 'rf'], default='ridgecv')
    parser.add_argument('--context-folds', type=int, default=5)
    parser.add_argument('--context-weight-cv', type=int, default=3)
    parser.add_argument('--context-cap-quantile', type=float, default=0.9)
    parser.add_argument('--context-max-weight', type=float, default=1.0)
    parser.add_argument('--skip-spo', action='store_true')
    parser.add_argument('--output', default='results/context_reweighting.csv')
    args = parser.parse_args()

    results_df = run_experiment(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print('wrote', output_path)
    print(results_df.groupby('algo')['tst_regret'].mean())


if __name__ == '__main__':
    main()
