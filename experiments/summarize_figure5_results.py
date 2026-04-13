import argparse

import pandas as pd


def summarize_best(df, algo_name):
    subset = df[df['algo'] == algo_name].copy()
    if subset.empty:
        return subset
    grouped = subset.groupby(['n_train', 'polykernel_degree', 'mixture_weight'], as_index=False)['tst_regret'].mean()
    best = grouped.sort_values('tst_regret').groupby(['n_train', 'polykernel_degree'], as_index=False).first()
    best['algo'] = algo_name + '_best'
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    fixed = df[df['algo'].isin(['LS', 'SPO'])].groupby(
        ['n_train', 'polykernel_degree', 'algo'], as_index=False
    )['tst_regret'].mean()

    summary = pd.concat(
        [
            fixed,
            summarize_best(df, 'reweight_LS'),
            summarize_best(df, 'context_reweight_LS'),
        ],
        ignore_index=True,
        sort=False
    )

    output = args.output or args.input_csv.replace('.csv', '_figure5_summary.csv')
    summary.to_csv(output, index=False)
    print('wrote', output)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
