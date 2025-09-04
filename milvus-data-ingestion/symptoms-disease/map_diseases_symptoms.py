"""
This script enriches disease data by mapping HPO (Human Phenotype Ontology) terms to diseases and expanding their symptom descriptions.

- Input:
  1. phenotype.hpoa: A file containing disease-to-symptom mappings.
  2. hpo_terms.json: A JSON file with HPO term details (names and definitions).
- Output:
  disease_kb_mapped.jsonl: A JSONL file with enriched disease data, including expanded symptom descriptions.

Steps:
1. Load HPO term details from the JSON file.
2. Read and process the phenotype.hpoa file to group diseases and their associated symptoms.
3. Map HPO IDs to their names and definitions, expanding symptom descriptions.
4. Save the enriched disease data to a JSONL file.

Usage:
Run this script directly to process the input files and generate the output file.
"""
# 03_expand_diseases.py
import json
import pandas as pd

HPOA_FILE = "../../data-files/symptoms-disease/phenotype.hpoa"
HPO_DICT_FILE = "../../data-files/symptoms-disease/hpo_terms.json"
OUT_FILE = "../../data-files/symptoms-disease/symptoms2disease.jsonl"

def main():
    with open(HPO_DICT_FILE, "r", encoding="utf-8") as f:
        hpo_terms = json.load(f)

    df = pd.read_csv(HPOA_FILE, sep="\t", comment="#", dtype=str).fillna("")

    grouped = df.groupby(["database_id", "disease_name"])["hpo_id"].apply(list).reset_index()

    count = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for _, row in grouped.iterrows():
            symptoms_expanded = []
            for hpo_id in row["hpo_id"]:
                if hpo_id in hpo_terms:
                    term = hpo_terms[hpo_id]
                    symptoms_expanded.append(f"{term['name']}: {term['definition']}")
                else:
                    symptoms_expanded.append(hpo_id)

            entry = {
                "disease_id": row["database_id"],
                "name": row["disease_name"],
                "symptoms": symptoms_expanded,
                "text": f"{row['disease_name']} is associated with symptoms: {', '.join(symptoms_expanded)}"
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"✅ Saved {count} enriched diseases to {OUT_FILE}")

if __name__ == "__main__":
    main()
