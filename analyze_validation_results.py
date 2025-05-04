"""
Analyze validation classification results: distributions, reports, and confusion matrices.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
import numpy as np


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'qid': str})
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['prediction'] = df['prediction'].astype(str).str.strip().str.lower()
    df.dropna(subset=['label', 'prediction'], inplace=True)
    return df


def print_nan_warning(df: pd.DataFrame) -> None:
    missing = df['prediction'].isna().sum()
    if missing:
        print(f"Warning: {missing} missing predictions.")


def print_distributions(rbc_df: pd.DataFrame,
                         hyb_df: pd.DataFrame) -> tuple:
    rbc_counts = rbc_df['prediction'].value_counts().sort_index()
    hyb_counts = hyb_df['prediction'].value_counts().sort_index()
    source_counts = hyb_df['source'].value_counts()

    print('Rule-based Prediction Counts:')
    print(rbc_counts.to_string(), end='\n\n')

    print('Hybrid Prediction Counts:')
    print(hyb_counts.to_string(), end='\n\n')

    print('Hybrid Source Breakdown:')
    print(source_counts.to_string(), end='\n')

    return rbc_counts, hyb_counts, source_counts


def print_classification_reports(rbc_df: pd.DataFrame,
                                 hyb_df: pd.DataFrame) -> None:
    print('Rule-based Classification Report:')
    print(classification_report(
        rbc_df['label'], rbc_df['prediction']
    ))
    print(f"Accuracy (Rule-based): {accuracy_score(
        rbc_df['label'], rbc_df['prediction']
    ):.3f}", end='\n\n')

    print('Hybrid Classification Report:')
    print(classification_report(
        hyb_df['label'], hyb_df['prediction']
    ))
    print(f"Accuracy (Hybrid): {accuracy_score(
        hyb_df['label'], hyb_df['prediction']
    ):.3f}", end='\n\n')


def plot_prediction_distribution(rbc_counts: pd.Series,
                                 hyb_counts: pd.Series) -> None:
    labels = sorted(set(rbc_counts.index) | set(hyb_counts.index))
    rbc_vals = [rbc_counts.get(lbl, 0) for lbl in labels]
    hyb_vals = [hyb_counts.get(lbl, 0) for lbl in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, rbc_vals, width, label='Rule-based')
    ax.bar(x + width/2, hyb_vals, width, label='Hybrid')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution: Rule-based vs Hybrid (Validation)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_source_breakdown(source_counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        source_counts,
        labels=source_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Hybrid Source Breakdown (Validation)')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(df: pd.DataFrame, title: str) -> None:
    labels = sorted(df['label'].unique())
    cm = confusion_matrix(
        df['label'], df['prediction'], labels=labels
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted label',
        ylabel='True label',
        title=title
    )
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='right'
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, cm[i, j],
                ha='center', va='center'
            )

    plt.tight_layout()
    plt.show()


def main(rbc_path: Path, hyb_path: Path) -> None:
    rbc_df = load_df(rbc_path)
    hyb_df = load_df(hyb_path)

    print_nan_warning(hyb_df)
    rbc_counts, hyb_counts, source_counts = print_distributions(
        rbc_df, hyb_df
    )
    print_classification_reports(rbc_df, hyb_df)
    plot_prediction_distribution(rbc_counts, hyb_counts)
    plot_source_breakdown(source_counts)
    plot_confusion_matrix(rbc_df, 'Confusion Matrix (Rule-based)')
    plot_confusion_matrix(hyb_df, 'Confusion Matrix (Hybrid)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze validation classification results'
    )
    parser.add_argument(
        '--rbc', type=Path,
        default=Path('results/validation_rules.csv'),
        help='Rule-based predictions CSV'
    )
    parser.add_argument(
        '--hyb', type=Path,
        default=Path('results/validation_fallback.csv'),
        help='Hybrid predictions CSV'
    )
    args = parser.parse_args()

    if not args.rbc.exists():
        parser.error(f'File not found: {args.rbc}')
    if not args.hyb.exists():
        parser.error(f'File not found: {args.hyb}')

    main(args.rbc, args.hyb)


