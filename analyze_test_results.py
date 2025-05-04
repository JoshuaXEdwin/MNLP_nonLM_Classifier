#!/usr/bin/env python3
"""
Analyze test classification results.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'qid': str})
    df['prediction'] = (
        df['prediction'].astype(str)
          .str.strip()
          .str.title()
    )
    if 'source' in df.columns:
        df['source'] = df['source'].fillna('QID')
    return df


def print_summary(rbc_counts: pd.Series, hyb_counts: pd.Series, source_counts: pd.Series) -> None:
    print('Rule-based prediction distribution:')
    print(rbc_counts.to_string(), end='\n\n')

    print('Hybrid prediction distribution:')
    print(hyb_counts.to_string(), end='\n\n')

    print('Prediction source breakdown:')
    print(source_counts.to_string())


def plot_label_distribution(rbc_counts: pd.Series, hyb_counts: pd.Series) -> None:
    labels = sorted(set(rbc_counts.index).union(hyb_counts.index))
    rbc_vals = [rbc_counts.get(label, 0) for label in labels]
    hyb_vals = [hyb_counts.get(label, 0) for label in labels]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rbc_vals, width, label='Rule-based')
    ax.bar([i + width for i in x], hyb_vals, width, label='Hybrid')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Count')
    ax.set_title('Predicted Label Distribution (Test Set)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_source_breakdown(source_counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    source_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    ax.set_title('Prediction Source Breakdown (Test Set)')
    plt.tight_layout()
    plt.show()


def main(rbc_path: Path, hyb_path: Path) -> None:
    rbc_df = load_predictions(rbc_path)
    hyb_df = load_predictions(hyb_path)

    rbc_counts = rbc_df['prediction'].value_counts()
    hyb_counts = hyb_df['prediction'].value_counts()
    source_counts = hyb_df['source'].value_counts()

    print_summary(rbc_counts, hyb_counts, source_counts)
    plot_label_distribution(rbc_counts, hyb_counts)
    plot_source_breakdown(source_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze test classification results.'
    )
    parser.add_argument(
        '--rbc',
        type=Path,
        default=Path('results/test_rules.csv'),
        help='Rule-based predictions CSV path'
    )
    parser.add_argument(
        '--hyb',
        type=Path,
        default=Path('results/hybrid_test_final_predicts.csv'),
        help='Hybrid predictions CSV path'
    )
    args = parser.parse_args()

    if not args.rbc.exists():
        parser.error(f'File not found: {args.rbc}')
    if not args.hyb.exists():
        parser.error(f'File not found: {args.hyb}')

    main(args.rbc, args.hyb)

