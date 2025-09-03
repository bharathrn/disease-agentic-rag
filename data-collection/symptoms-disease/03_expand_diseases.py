# 03_expand_diseases.py
import json
import pandas as pd

HPOA_FILE = "../data-files/phenotype.hpoa"
HPO_DICT_FILE = "../data-files/hpo_terms.json"
OUT_FILE = "../data-files/disease_kb_mapped.jsonl"

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
