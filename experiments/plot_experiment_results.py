import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    'LS': 'LS',
    'reweight_LS': 'Task-Loss Reweight',
    'context_reweight_LS': 'Context Reweight',
    'SPO': 'SPO+',
}

METHOD_COLORS = {
    'LS': '#2a5c8a',
    'reweight_LS': '#d95f02',
    'context_reweight_LS': '#1b9e77',
    'SPO': '#7b3294',
}

METHOD_STYLES = {
    'LS': '-',
    'reweight_LS': '-',
    'context_reweight_LS': '--',
    'SPO': '-.',
}


def aggregate_results(df):
    grouped = df.groupby(
        ['polykernel_degree', 'n_train', 'algo'],
        as_index=False
    )['tst_regret'].agg(['mean', 'std', 'count']).reset_index()
    grouped['stderr'] = grouped['std'].fillna(0.0) / grouped['count'].clip(lower=1).pow(0.5)
    return grouped


def add_endpoint_label(ax, x_values, y_values, label, color):
    if len(x_values) == 0:
        return
    ax.annotate(
        label,
        xy=(x_values[-1], y_values[-1]),
        xytext=(8, 0),
        textcoords='offset points',
        color=color,
        fontsize=9.5,
        va='center',
        fontweight='medium',
    )


def plot_results(df, output_path, title):
    summary = aggregate_results(df)
    degrees = sorted(summary['polykernel_degree'].unique())
    methods = [algo for algo in ['LS', 'reweight_LS', 'context_reweight_LS', 'SPO']
               if algo in summary['algo'].unique()]

    fig, axes = plt.subplots(
        1,
        len(degrees),
        figsize=(3.8 * len(degrees), 3.8),
        sharex=True,
        constrained_layout=True,
    )
    if len(degrees) == 1:
        axes = [axes]

    for ax, degree in zip(axes, degrees):
        degree_df = summary[summary['polykernel_degree'] == degree]
        for algo in methods:
            algo_df = degree_df[degree_df['algo'] == algo].sort_values('n_train')
            if algo_df.empty:
                continue
            x_values = algo_df['n_train'].to_list()
            y_values = algo_df['mean'].to_list()
            stderr_values = algo_df['stderr'].to_list()
            color = METHOD_COLORS[algo]
            ax.plot(
                x_values,
                y_values,
                marker='o',
                linewidth=2,
                color=color,
                linestyle=METHOD_STYLES[algo],
                label=METHOD_LABELS[algo],
            )
            lower = [y - e for y, e in zip(y_values, stderr_values)]
            upper = [y + e for y, e in zip(y_values, stderr_values)]
            ax.fill_between(x_values, lower, upper, alpha=0.18, color=color)
            add_endpoint_label(ax, x_values, y_values, METHOD_LABELS[algo], color)

        ax.set_title('Linear regression, poly degree, %s' % degree)
        ax.set_xlabel('Train Size')
        ax.set_ylabel('Mean Test Regret')
        ax.set_xticks(sorted(degree_df['n_train'].unique()))
        ax.grid(axis='y', alpha=0.25, linewidth=0.8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, fontsize=13.5, y=1.14)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--title', default='Experiment Results')
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output or input_path.with_suffix('.png'))

    df = pd.read_csv(input_path)
    plot_results(df, output_path, args.title)
    print('wrote', output_path)


if __name__ == '__main__':
    main()
