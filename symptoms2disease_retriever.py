# 03_query_milvus_aggregate.py
"""
Query Milvus with a natural language symptom description,
retrieve top-K matching chunks and then aggregate scores by disease_id
to produce disease-ranked results.

If index doesn't exist, create HNSW index automatically.
"""

from typing import List, Dict
from collections import defaultdict
import numpy as np
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# CONFIG (keep consistent with ingestion)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "disease_kb_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384  # must match your embedding model

TOP_K_CHUNKS = 40     # how many chunk-level hits to retrieve
TOP_N_DISEASES = 2    # how many diseases to return
TOP_M_CHUNKS_PER_DISEASE = 4  # how many top chunks to keep per disease for context

# Connect
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Load collection
collection = Collection(COLLECTION_NAME)

# Ensure index exists
if not collection.has_index():
    print(f"[INFO] No index found for '{COLLECTION_NAME}'. Creating HNSW index...")
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",   # inner product (use COSINE if you didn’t normalize vectors)
            "params": {"M": 48, "efConstruction": 200}
        }
    )
    print("[INFO] Index created successfully.")

# Load into memory
collection.load()

# Embedder
embedder = SentenceTransformer(EMBEDDING_MODEL)

def norm_vec(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def query_and_aggregate(query_text: str, top_k_chunks: int = TOP_K_CHUNKS):
    # Encode query
    q_vec = embedder.encode([query_text], convert_to_numpy=True)[0]
    q_vec = norm_vec(q_vec).astype(np.float32).tolist()

    # Run search
    results = collection.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},  # ef controls recall
        limit=top_k_chunks,
        output_fields=["disease_id", "disease_name", "chunk_index", "chunk_text"],
    )

    hits = results[0]
    agg = defaultdict(lambda: {"name": None, "scores": [], "chunks": []})

    # Aggregate by disease
    for hit in hits:
        score = float(hit.distance)
        ent = hit.entity
        did = ent.get("disease_id")
        name = ent.get("disease_name")
        chunk_idx = ent.get("chunk_index")
        text_snippet = ent.get("chunk_text")

        agg[did]["name"] = name
        agg[did]["scores"].append(score)
        agg[did]["chunks"].append((score, chunk_idx, text_snippet))

    disease_list = []
    for did, info in agg.items():
        # Sort chunks by score
        info["chunks"].sort(key=lambda x: x[0], reverse=True)
        sorted_scores = sorted(info["scores"], reverse=True)
        top_scores = sorted_scores[: min(5, len(sorted_scores))]
        agg_score = float(np.mean(top_scores))
        disease_list.append((did, info["name"], agg_score, info["chunks"]))

    # Sort by aggregated score
    disease_list.sort(key=lambda x: x[2], reverse=True)

    # Prepare final output
    results_out = []
    for did, name, score, chunks in disease_list[:TOP_N_DISEASES]:
        top_chunks_texts = [c[2] for c in chunks[:TOP_M_CHUNKS_PER_DISEASE]]
        combined_context = "\n---\n".join(top_chunks_texts)
        results_out.append({
            "disease_id": did,
            "name": name,
            "score": score,
            "context": combined_context,
            "top_chunks": [
                {"score": float(c[0]), "chunk_index": int(c[1]), "text": c[2]}
                for c in chunks[:TOP_M_CHUNKS_PER_DISEASE]
            ],
        })

    return results_out

if __name__ == "__main__":
    q = input("Describe symptoms: ").strip()
    out = query_and_aggregate(q)
    print("\n=== Top disease candidates ===\n")
    for i, item in enumerate(out, 1):
        print(f"{i}. {item['name']} ({item['disease_id']}) — agg_score={item['score']:.4f}")
        print("   Top chunks:")
        for c in item["top_chunks"]:
            print(f"     - score={c['score']:.4f} idx={c['chunk_index']} snippet={c['text'][:200]}...")
        print("   Combined context (first 300 chars):")
        print(item["context"][:300].replace("\n", " ") + "...\n")
