# pip install sentence-transformers pymilvus

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import json

# --------------------------
# Config
# --------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # Embedding model
EMB_DIM = 384                           # Dim of above model
COLLECTION_NAME = "disease_treatments"

# --------------------------
# Load model
# --------------------------
model = SentenceTransformer(MODEL_NAME)

# --------------------------
# Connect to Milvus
# --------------------------
connections.connect(alias="default", host="127.0.0.1", port="19530")

# --------------------------
# Drop collection if exists
# --------------------------
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# --------------------------
# Define schema
# --------------------------
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="disease_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM),
    FieldSchema(name="treatments", dtype=DataType.VARCHAR, max_length=5000),
]
schema = CollectionSchema(fields, description="Disease → Treatments mapping")

collection = Collection(name=COLLECTION_NAME, schema=schema)

# --------------------------
# Load dataset (your JSON file)
# --------------------------
with open("../../data-files/disease-treatement/disease2treatements.json", "r") as f:
    data = json.load(f)

# --------------------------
# Build embeddings (only disease name)
# --------------------------
names = [d["name"] for d in data]
embeddings = model.encode(names, normalize_embeddings=True)

# --------------------------
# Insert into Milvus
# --------------------------
entities = [
    [d["disease_id"] for d in data],
    names,
    embeddings.tolist(),
    [", ".join(d["treatments"]) for d in data],
]

collection.insert(entities)
collection.flush()

print(f"✅ Inserted {len(data)} diseases into Milvus")
