"""
Query Milvus for treatments given a disease name.
Each disease is one row (no chunking needed).
"""

from typing import List, Dict
import numpy as np
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# --------------------
# CONFIG
# --------------------
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "disease_treatments"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384  # must match ingestion

TOP_K = 2   # how many diseases to return

# --------------------
# Connect + Load
# --------------------
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)

# Create index if not exists
if not collection.has_index():
    print(f"[INFO] No index found for '{COLLECTION_NAME}'. Creating HNSW index...")
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",   # use IP since we normalized embeddings
            "params": {"M": 48, "efConstruction": 200}
        }
    )
    print("[INFO] Index created successfully.")

# Load into memory
collection.load()

# --------------------
# Embedder
# --------------------
embedder = SentenceTransformer(EMBEDDING_MODEL)

def norm_vec(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# --------------------
# Query function
# --------------------
def retrieve_treatments(query: str, top_k: int = TOP_K) -> List[Dict]:
    # Encode query
    q_vec = embedder.encode([query], convert_to_numpy=True)[0]
    q_vec = norm_vec(q_vec).astype(np.float32).tolist()

    # Run vector search
    results = collection.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["disease_id", "name", "treatments"],
    )

    hits = results[0]
    output = []
    for hit in hits:
        ent = hit.entity
        output.append({
            "disease_id": ent.get("disease_id"),
            "name": ent.get("name"),
            "treatments": ent.get("treatments").split(", "),
            "score": float(hit.distance)
        })
    return output


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    q = input("Enter disease name: ").strip()
    out = retrieve_treatments(q, top_k=1)

    print("\n=== Top treatment candidates ===\n")
    for i, item in enumerate(out, 1):
        print(f"{i}. {item['name']} ({item['disease_id']}) — score={item['score']:.4f}")
        print("   Treatments:")
        for t in item["treatments"]:
            print(f"     - {t}")
        print()
