"""
This script processes the Human Phenotype Ontology (HPO) file to extract terms and their details, saving them in a JSON file.

- Input: hp.obo file (ontology file containing HPO terms and definitions).
- Output: hpo_terms.json file with HPO term IDs, names, and definitions.

Steps:
1. Load the HPO ontology file using the `pronto` library.
2. Extract terms starting with "HP:", along with their names and definitions.
3. Save the extracted data to a JSON file in a structured format.

Usage:
Run this script directly to process the input file and generate the output file.
"""

import pronto
import json

ONTO_PATH = "../../data-files/symptoms-disease/hp.obo"
OUT_JSON = "../../data-files/symptoms-disease/hpo_terms.json"

def main():
    ontology = pronto.Ontology(ONTO_PATH)
    hpo_dict = {}

    for term in ontology.terms():
        if term.id.startswith("HP:"):
            hpo_dict[term.id] = {
                "name": term.name,
                "definition": term.definition
            }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(hpo_dict, f, indent=2)
    print(f"Saved {len(hpo_dict)} HPO terms to {OUT_JSON}")

if __name__ == "__main__":
    main()
