# Loading Data to Milvus

This directory contains scripts for ingesting data into the Milvus vector database.

## Steps to Load Data

1. **Install Dependencies**:
   - Ensure all required Python packages are installed. Refer to the `pyproject.toml` file for dependencies.

2. **Configure Milvus**:
   - Set up the Milvus server and ensure it is running.

3. **Run Ingestion Scripts**:
   - For symptoms-to-diseases data load to milvus:
     ```bash
     python symptoms-disease/ingest-symptoms-diseases.py
     ```

   - For diseases-to-treatments data:

   - Data Extraction:
     ```bash
     python disease-treatement/map_symptoms_explanations.py
     python disease-treatement/map_diseases_symptoms.py
     ```
    - Data Load to Milvus:
     ```bash
     python disease-treatement/ingest-diseases-treatements.py
     ```

4. **Verify Data**:
   - Use Milvus client tools to verify that the data has been ingested correctly.

---

Refer to the respective subdirectories for detailed scripts and workflows.
