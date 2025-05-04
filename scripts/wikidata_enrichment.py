"""
Enrich a list of Wikidata QIDs with cultural properties and save results.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

# Ensure parent project directory is on PYTHONPATH for constants import
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.constants import WIKIDATA_PROPERTIES  # type: ignore


def fetch_wikidata_entity(qid: str, session: requests.Session) -> dict | None:
    """
    Retrieve the Wikidata JSON for a given QID.
    Returns the entity dict or None on failure.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("entities", {}).get(qid)
    except requests.RequestException:
        return None


def extract_cultural_properties(entity: dict | None, qid: str) -> dict:
    """
    Extracts specified properties from a Wikidata entity.
    Returns a dict with 'qid' and list of values per property.
    """
    claims = entity.get("claims", {}) if entity else {}
    result = {"qid": qid}

    for prop_name, pid in WIKIDATA_PROPERTIES.items():
        values: list[str] = []
        for entry in claims.get(pid, []):
            mainsnak = entry.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict) and "id" in value:
                values.append(value["id"].lower())
        result[prop_name] = values

    return result


def enrich_qids(input_file: Path, output_file: Path, delay: float) -> None:
    """
    Main enrichment pipeline: reads input CSV with 'item' column (QIDs),
    queries Wikidata, extracts properties, and writes output CSV.
    """
    df = pd.read_csv(input_file)
    qids = df["item"].astype(str).str.split("/").str[-1]

    session = requests.Session()
    records: list[dict] = []

    for qid in tqdm(qids, desc="Enriching QIDs", unit="qid"):
        entity = fetch_wikidata_entity(qid, session)
        record = extract_cultural_properties(entity, qid)
        records.append(record)
        time.sleep(delay)

    enriched_df = pd.DataFrame.from_records(records)
    enriched_df.to_csv(output_file, index=False)
    print(f"Enriched {len(records)} QIDs -> {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikidata enrichment for cultural properties."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV containing 'item' column with QIDs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save enriched output CSV."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between API calls (rate limiting)."
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enrich_qids(args.input, args.output, args.delay)


if __name__ == "__main__":
    main()



