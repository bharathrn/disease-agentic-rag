# 02_ingest_milvus_chunked.py
"""
Ingest a JSONL disease KB where each line is a disease dict with fields:
  - disease_id
  - name
  - text   (long descriptions / symptom paragraphs)

This script:
 - tokenizes using a HuggingFace tokenizer (same family as embedder)
 - creates sliding-window token chunks with overlap (token-aware)
 - encodes each chunk to embeddings (batching)
 - normalizes embeddings (L2) so IP ~ cosine
 - inserts rows into Milvus, where each row = one chunk
"""

import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# CONFIG
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "disease_kb_chunks"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dim
TOKENIZER_NAME = EMBEDDING_MODEL
DIM = 384

# chunking params (tokens)
MAX_TOKENS = 256        # tokens per chunk (safe for CPU)
OVERLAP_TOKENS = 48     # overlap between consecutive chunks
BATCH_SIZE = 256        # number of chunks to encode/insert per batch
CHUNK_TEXT_MAX_LENGTH = 65535  # VARCHAR max_length in Milvus (set large)

INPUT_JSONL = "./data-files/disease_kb_mapped.jsonl"   # file you showed
assert Path(INPUT_JSONL).exists(), f"Put your JSONL at: {INPUT_JSONL}"

# Connect
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Schema: chunk_id (auto primary), embedding, disease_id, disease_name, chunk_index, chunk_text
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema(name="disease_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="disease_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=CHUNK_TEXT_MAX_LENGTH),
]
schema = CollectionSchema(fields, description="Chunked disease KB")
# Drop and recreate
if utility.has_collection(COLLECTION_NAME):
    print("Dropping existing collection...")
    utility.drop_collection(COLLECTION_NAME)

collection = Collection(COLLECTION_NAME, schema)
print("Created collection:", COLLECTION_NAME)

# Models
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def token_chunker(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS):
    """
    Token-based sliding window chunker.
    Yields (chunk_text, chunk_index).
    """
    if not text:
        return
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return

    stride = max_tokens - overlap
    # if text shorter than a chunk, yield it once
    if len(token_ids) <= max_tokens:
        chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        yield chunk_text, 0
        return

    idx = 0
    chunk_idx = 0
    while idx < len(token_ids):
        window = token_ids[idx: idx + max_tokens]
        chunk_text = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        yield chunk_text, chunk_idx
        chunk_idx += 1
        idx += stride

def norm_np(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Ingest
def ingest(jsonl_path: str):
    embeddings_batch = []
    disease_ids_batch = []
    disease_names_batch = []
    chunk_indices_batch = []
    chunk_texts_batch = []
    total_chunks = 0

    def flush_batch():
        nonlocal embeddings_batch, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch, total_chunks
        if not embeddings_batch:
            return
        # insert: since chunk_id is auto_id, we omit it from insert lists.
        collection.insert([embeddings_batch, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch])
        collection.flush()
        total_chunks += len(embeddings_batch)
        print(f"Inserted batch of {len(embeddings_batch)} chunks (total {total_chunks})")
        embeddings_batch, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch = [], [], [], [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            disease_id = obj.get("disease_id", "UNKNOWN")
            disease_name = obj.get("name", "") or ""
            text = obj.get("text", "") or obj.get("description", "") or " ".join(obj.get("symptoms", []))

            # create chunks
            for chunk_text, chunk_index in token_chunker(text, MAX_TOKENS, OVERLAP_TOKENS):
                # guard chunk_text length to Milvus varchar max
                if len(chunk_text) > CHUNK_TEXT_MAX_LENGTH:
                    chunk_text = chunk_text[:CHUNK_TEXT_MAX_LENGTH]

                # encode & normalize (we'll batch encode below for efficiency)
                embeddings_batch.append(chunk_text)    # temp store text; will encode a batch of texts
                # store metadata parallel lists
                disease_ids_batch.append(disease_id)
                disease_names_batch.append(disease_name)
                chunk_indices_batch.append(chunk_index)
                chunk_texts_batch.append(chunk_text)

                # if we reached batch size, encode the text-batch to vectors and replace strings with vectors
                if len(embeddings_batch) >= BATCH_SIZE:
                    # embeddings_batch currently holds chunk_texts -> encode and normalize
                    texts_to_encode = embeddings_batch
                    vecs = embedder.encode(texts_to_encode, convert_to_numpy=True, show_progress_bar=False)
                    # normalize each vector inplace
                    vecs = np.vstack([norm_np(v) for v in vecs]).astype(np.float32)
                    # convert to python list lists (Milvus expects lists)
                    vecs_list = vecs.tolist()

                    # perform insert with vector lists
                    collection.insert([vecs_list, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch])
                    collection.flush()
                    total_chunks += len(vecs_list)
                    print(f"Inserted batch of {len(vecs_list)} chunks (total {total_chunks})")

                    # reset
                    embeddings_batch, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch = [], [], [], [], []

    # flush final small batch
    if embeddings_batch:
        texts_to_encode = embeddings_batch
        vecs = embedder.encode(texts_to_encode, convert_to_numpy=True, show_progress_bar=False)
        vecs = np.vstack([norm_np(v) for v in vecs]).astype(np.float32)
        vecs_list = vecs.tolist()
        collection.insert([vecs_list, disease_ids_batch, disease_names_batch, chunk_indices_batch, chunk_texts_batch])
        collection.flush()
        total_chunks += len(vecs_list)
        print(f"Inserted final batch of {len(vecs_list)} chunks (total {total_chunks})")

    # Create index and load collection for searching
    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    print("✅ Done ingesting. Total chunks:", total_chunks)

if __name__ == "__main__":
    ingest(INPUT_JSONL)
