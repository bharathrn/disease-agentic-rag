"""
Query Milvus for treatments given a disease name.
Each disease is one row (no chunking needed).
"""

from typing import List, Dict
import numpy as np
from pymilvus import Collection
from utils.config import MILVUS_HOST, MILVUS_PORT, DIM
from utils.vector_utils import norm_vec
from utils.milvus_utils import connect_to_milvus, get_or_create_collection
from utils.embedding_utils import load_embedder
from utils.constants import TREATMENT_EMBEDDING_MODEL, TREATMENT_TOP_K

# --------------------
# CONFIG
# --------------------
COLLECTION_NAME = "disease_treatments"

EMBEDDING_MODEL = TREATMENT_EMBEDDING_MODEL
TOP_K = TREATMENT_TOP_K   # how many diseases to return

# --------------------
# Connect + Load
# --------------------
connect_to_milvus(MILVUS_HOST, MILVUS_PORT)

# Load collection
collection = get_or_create_collection(
    collection_name="disease_treatments",
    dim=DIM,
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 48, "efConstruction": 200}
    }
)

# Load embedder
embedder = load_embedder(EMBEDDING_MODEL)

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
