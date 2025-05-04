#!/usr/bin/env python3
"""
Apply ML fallback to rule-based test predictions and save hybrid results.
"""
import argparse
from pathlib import Path

import pandas as pd
import joblib


def load_raw_text(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df['qid'] = df['item'].str.extract(r'(Q\d+)', expand=False)
    return df[['qid', 'name', 'description']]


def merge_predictions_with_text(preds: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    df = preds.merge(raw, on='qid', how='left')
    df['text'] = (df['name'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
    return df


def apply_ml_fallback(df: pd.DataFrame, model) -> pd.DataFrame:
    mask = (
        df['prediction'].fillna('').str.lower() == 'cultural agnostic'
    ) & df['text'].str.len().gt(0)
    if mask.any():
        fallback_preds = model.predict(df.loc[mask, 'text'])
        df.loc[mask, 'prediction'] = (
            pd.Series(fallback_preds, index=df.loc[mask].index)
              .str.title()
        )
        df.loc[mask, 'source'] = 'ML'
    df['source'] = df['source'].fillna('QID')
    missing = df['prediction'].isna().sum()
    if missing:
        print(f"Warning: {missing} missing predictions after fallback.")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Apply ML fallback to test rule-based predictions.'
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path('results/test_rules.csv'),
        help='Rule-based predictions CSV'
    )
    parser.add_argument(
        '-r', '--raw',
        type=Path,
        default=Path('data/test_unlabeled.csv'),
        help='Raw test dataset CSV'
    )
    parser.add_argument(
        '-m', '--model',
        type=Path,
        default=Path('models/tfidf_fallback_model.pkl'),
        help='Serialized fallback model file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('results/hybrid_test_final_predicts.csv'),
        help='Output CSV for hybrid test final predictions'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load and merge
    preds = pd.read_csv(args.input, dtype={'qid': str})
    raw   = load_raw_text(args.raw)
    df    = merge_predictions_with_text(preds, raw)

    # Load model and apply fallback
    if not args.model.exists():
        parser.error(f'Model file not found: {args.model}')
    model = joblib.load(args.model)

    df = apply_ml_fallback(df, model)
    df[['qid', 'prediction', 'source']].to_csv(args.output, index=False)
    print(f'Hybrid test predictions saved to: {args.output}')


if __name__ == '__main__':
    main()






