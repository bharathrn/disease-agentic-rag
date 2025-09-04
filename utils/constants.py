# Constants for RAG applications

# Symptoms to Disease Retriever Constants
SYMPTOMS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SYMPTOMS_TOP_K_CHUNKS = 40
SYMPTOMS_TOP_N_DISEASES = 2
SYMPTOMS_TOP_M_CHUNKS_PER_DISEASE = 4

# Disease to Treatment Retriever Constants
TREATMENT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TREATMENT_TOP_K = 1
