from sentence_transformers import SentenceTransformer

def load_embedder(model_name: str):
    """Load a SentenceTransformer model."""
    return SentenceTransformer(model_name)
