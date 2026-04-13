import argparse
from pathlib import Path

import pandas as pd


DIAGNOSTIC_COLUMNS = [
    'holdout_mse',
    'holdout_regret',
    'pilot_regret_weighted_holdout_mse',
    'cv_train_mse',
    'cv_train_weighted_mse',
]

METHODS = ['reweight_LS', 'context_reweight_LS']


def select_rows(subset, value_column):
    ordered = subset.sort_values([value_column, 'mixture_weight']).copy()
    keys = ['task_id', 'algo']
    return ordered.groupby(keys, as_index=False).first()


def summarize_method(df, algo_name):
    subset = df[df['algo'] == algo_name].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    oracle = select_rows(subset, 'tst_regret').rename(
        columns={
            'mixture_weight': 'oracle_mu',
            'tst_regret': 'oracle_tst_regret',
        }
    )
    oracle = oracle[['task_id', 'algo', 'oracle_mu', 'oracle_tst_regret']]

    summary_rows = []
    detail_frames = []
    for diagnostic in DIAGNOSTIC_COLUMNS:
        available = subset.dropna(subset=[diagnostic]).copy()
        if available.empty:
            continue
        selected = select_rows(available, diagnostic).rename(
            columns={
                'mixture_weight': 'selected_mu',
                'tst_regret': 'selected_tst_regret',
                diagnostic: 'selected_diagnostic_value',
            }
        )
        detail = selected.merge(oracle, on=['task_id', 'algo'], how='inner')
        detail['diagnostic'] = diagnostic
        detail['mu_exact_match'] = detail['selected_mu'] == detail['oracle_mu']
        detail['tst_regret_gap'] = detail['selected_tst_regret'] - detail['oracle_tst_regret']
        detail_frames.append(
            detail[
                [
                    'task_id',
                    'algo',
                    'diagnostic',
                    'selected_mu',
                    'oracle_mu',
                    'selected_diagnostic_value',
                    'selected_tst_regret',
                    'oracle_tst_regret',
                    'tst_regret_gap',
                    'mu_exact_match',
                ]
            ]
        )

        summary_rows.append(
            {
                'algo': algo_name,
                'diagnostic': diagnostic,
                'num_tasks': len(detail),
                'exact_match_rate': detail['mu_exact_match'].mean(),
                'mean_selected_mu': detail['selected_mu'].mean(),
                'mean_oracle_mu': detail['oracle_mu'].mean(),
                'mean_selected_tst_regret': detail['selected_tst_regret'].mean(),
                'mean_oracle_tst_regret': detail['oracle_tst_regret'].mean(),
                'mean_tst_regret_gap': detail['tst_regret_gap'].mean(),
                'median_tst_regret_gap': detail['tst_regret_gap'].median(),
            }
        )

    summary = pd.DataFrame.from_records(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--details-output', default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    summaries = []
    details = []
    for algo_name in METHODS:
        summary, detail = summarize_method(df, algo_name)
        if not summary.empty:
            summaries.append(summary)
        if not detail.empty:
            details.append(detail)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    details_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    input_path = Path(args.input_csv)
    output = Path(args.output or input_path.with_name(input_path.stem + '_mu_selection_summary.csv'))
    details_output = Path(
        args.details_output or input_path.with_name(input_path.stem + '_mu_selection_details.csv')
    )

    summary_df.to_csv(output, index=False)
    details_df.to_csv(details_output, index=False)
    print('wrote', output)
    print('wrote', details_output)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
