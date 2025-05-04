#!/usr/bin/env python3
"""
Run rule-based cultural classification on the validation set and save predictions.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_golden_rules(instance_path: Path, culture_path: Path, heritage_path: Path):
    """
    Load golden QID-to-label mappings for instance, culture, and heritage.
    """
    instance_map = pd.read_csv(instance_path).set_index('qid')['label'].to_dict()
    culture_map = pd.read_csv(culture_path).set_index('qid')['label'].to_dict()
    heritage_map = pd.read_csv(heritage_path).set_index('qid')['label'].to_dict()
    return instance_map, culture_map, heritage_map


def rule_based_classifier(row, instance_map, culture_map, heritage_map):
    """
    Apply rule-based lookup: heritage → culture → instance.
    Returns (prediction, source).
    """
    for qid in row.get('heritage_status', []):
        label = heritage_map.get(qid.lower())
        if label:
            return label, 'QID'
    for qid in row.get('part_of_culture', []):
        label = culture_map.get(qid.lower())
        if label:
            return label, 'QID'
    for qid in row.get('instance_of', []):
        label = instance_map.get(qid.lower())
        if label:
            return label, 'QID'
    return 'Cultural Agnostic', 'Agnostic'


def main():
    parser = argparse.ArgumentParser(
        description='Rule-based classification on validation data'
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path('data/validation_enriched_with_labels.csv'),
        help='Path to enriched validation CSV'
    )
    parser.add_argument(
        '--golden-instance',
        type=Path,
        default=Path('data/golden_instance_qids.csv'),
        help='Golden instance QIDs CSV'
    )
    parser.add_argument(
        '--golden-culture',
        type=Path,
        default=Path('data/golden_culture_qids.csv'),
        help='Golden culture QIDs CSV'
    )
    parser.add_argument(
        '--golden-heritage',
        type=Path,
        default=Path('data/golden_heritage_qids.csv'),
        help='Golden heritage QIDs CSV'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('results'),
        help='Directory to save rule-based validation predictions'
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / 'validation_rules.csv'

    df = pd.read_csv(args.input, dtype={'qid': str})
    list_fields = [
        'country_of_origin', 'country', 'located_in',
        'part_of_culture', 'instance_of', 'heritage_status'
    ]
    for field in list_fields:
        df[field] = df[field].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

    instance_map, culture_map, heritage_map = load_golden_rules(
        args.golden_instance,
        args.golden_culture,
        args.golden_heritage
    )

    tqdm.pandas(desc='Applying rule-based classifier to validation set')
    results = df.progress_apply(
        lambda row: pd.Series(
            rule_based_classifier(row, instance_map, culture_map, heritage_map)
        ),
        axis=1
    )
    df[['prediction', 'source']] = results
    df['prediction'] = df['prediction'].str.title()

    df[['qid', 'label', 'prediction', 'source']].to_csv(output_file, index=False)
    print(f'Validation rule-based predictions saved to: {output_file}')

if __name__ == '__main__':
    main()

