"""
This script processes the HPOA (Human Phenotype Ontology Annotation) file to generate a JSONL file containing diseases and their associated symptoms.

- Input: phenotype.hpoa file (tab-separated values with metadata lines starting with '#').
- Output: disease_kb.jsonl file with disease IDs, names, symptoms (HPO IDs), and additional metadata.

Steps:
1. Load the HPOA file, skipping metadata lines.
2. Group data by disease ID and name, collecting associated symptoms.
3. Write the processed data to a JSONL file, including additional metadata fields.

Usage:
Run this script directly to process the input file and generate the output file.
"""
# 01_load_hpoa.py
import pandas as pd
import json

HPOA_FILE = "../../data-files/symptoms-disease/phenotype.hpoa"
OUT_FILE = "../../data-files/symptoms-disease/disease_kb.jsonl"

def main():
    # Load TSV, skip the metadata lines starting with '#'
    df = pd.read_csv(HPOA_FILE, sep="\t", comment="#", dtype=str).fillna("")

    # Group by disease_id and collect symptoms
    grouped = df.groupby(["database_id", "disease_name"])["hpo_id"].apply(list).reset_index()

    count = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for _, row in grouped.iterrows():
            entry = {
                "disease_id": row["database_id"],
                "name": row["disease_name"],
                "symptoms": row["hpo_id"],   # list of HPO IDs
                "short_description": f"HPO-linked disorder: {row['disease_name']}",
                "prevalence": "unknown"
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"✅ Saved {count} diseases with symptoms to {OUT_FILE}")

if __name__ == "__main__":
    main()
