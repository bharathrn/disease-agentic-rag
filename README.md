# Disease-Agentic-RAG

This repository provides a comprehensive system for mapping symptoms to diseases and diseases to treatments. It leverages data ingestion, processing, and retrieval workflows, along with a Milvus vector database for efficient data storage and querying. Below is an overview of the application flow and its components.

## Highlights

- **Vibe Coding**: All the code in this repository was written using Vibe Coding with minimal intervention by me.
- **Purpose**: This repository was created to revise concepts for interview preparation.

## Demo

[Download and watch the demo](./demos/app-demo.mp4)


## Application Flow

1. **Data Collection**:
   - Collect data for symptoms-to-diseases mapping.
   - Collect data for diseases-to-treatments mapping.

2. **Data Ingestion**:
   - Process and ingest the collected data into the Milvus vector database.

3. **Agent Workflow**:
   - Use the agent to query the database and retrieve relevant information.

4. **Running the Application**:
   - Start the application and interact with the system to retrieve insights.

## Components

- **Data Files**: Contains raw data files for symptoms, diseases, and treatments.
- **Milvus Data Ingestion**: Scripts for ingesting data into the Milvus database.
- **Agent**: The core logic for querying and retrieving data.
- **Static Files**: Contains static assets like the `index.html` file.
- **Utils**: Utility scripts for configurations, constants, and vector operations.

---

For detailed instructions, refer to the respective README files in the repository.
