import pronto
import json

ONTO_PATH = "hp.obo"
OUT_JSON = "hpo_terms.json"

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
