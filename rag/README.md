# Agent

This directory contains the core logic for querying and retrieving data from the Milvus vector database.

## Components

- `disease2treatement_retriever.py`: Retrieves treatments for a given disease.
- `symptoms2disease_retriever.py`: Retrieves diseases for given symptoms.

## Workflow

1. **Input Query**:
   - Accepts user input for symptoms or diseases.

2. **Vectorization**:
   - Converts input queries into vector representations using embedding utilities.

3. **Milvus Query**:
   - Queries the Milvus database to retrieve relevant results.

4. **Output Results**:
   - Returns the retrieved results to the user.

---

Refer to the `utils` directory for supporting utilities.
