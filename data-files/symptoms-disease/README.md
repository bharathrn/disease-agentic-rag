# Symptoms-to-Diseases Data Collection

This directory contains data files and scripts for mapping symptoms to diseases.

## Data Files

- `disease_kb.jsonl`: Knowledge base for diseases.
- `hp.obo`: Ontology file for Human Phenotype Ontology (HPO).
- `hpo_terms.json`: JSON file containing HPO terms.
- `phenotype.hpoa`: Phenotype annotations.
- `symptoms2disease.jsonl`: Mapping of symptoms to diseases.

## Data Collection Workflow

1. **Source Identification**:
   - Identify reliable sources for symptoms and diseases data.

2. **Data Extraction**:
   - Extract data from the identified sources.

3. **Data Cleaning**:
   - Remove duplicates and inconsistencies.

4. **Data Formatting**:
   - Convert data into JSONL format for ingestion.

5. **Validation**:
   - Validate the data against known standards.

---

Refer to the `milvus-data-ingestion/symptoms-disease` directory for ingestion scripts.
