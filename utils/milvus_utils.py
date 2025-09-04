from pymilvus import connections, Collection

def connect_to_milvus(host: str, port: str, alias: str = "default"):
    """Connect to Milvus server."""
    connections.connect(alias=alias, host=host, port=port)

def get_or_create_collection(collection_name: str, dim: int, index_params: dict):
    """Get or create a Milvus collection with the specified index."""
    collection = Collection(collection_name)
    if not collection.has_index():
        print(f"[INFO] No index found for '{collection_name}'. Creating index...")
        collection.create_index(field_name="embedding", index_params=index_params)
        print("[INFO] Index created successfully.")
    collection.load()
    return collection
