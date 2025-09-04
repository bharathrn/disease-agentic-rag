# How to Run the Application

Follow these steps to set up and run the application.

## Prerequisites

- Python 3.10 or higher
- Milvus vector database
- Required Python packages (refer to `pyproject.toml`)

## Steps to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Milvus**:
   - Ensure the Milvus server is running.

3. **Ingest Data**:
   - Run the ingestion scripts to load data into Milvus.

4. **Run the Application**:

   - Make sure data is ingested to Milvus.
        - data for symptoms to disease mapping
        - data for disease to treatements mapping.
        Check scripts in milvus data ingestion folder.
   - Make sure your milvus is running in local instance.
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Interact with the Application**:
   - Use the provided interface or API to query the system.

## Hosting Milvus Locally

To host Milvus locally, follow these steps:

1. **Clone the Milvus Repository**:
   ```bash
   git clone https://github.com/milvus-io/milvus.git
   ```

2. **Navigate to the Docker Standalone Deployment Directory**:
   ```bash
   cd milvus/deployments/docker/standalone
   ```

3. **Start Milvus Using Docker Compose**:
   ```bash
   docker-compose up -d
   ```

Ensure that the Milvus server is running before proceeding with data ingestion and application execution.

---

Refer to the `README.md` file for an overview of the repository.
