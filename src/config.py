import os
from qdrant_client import QdrantClient
from fastembed import ImageEmbedding, TextEmbedding

def get_qdrant_client():
    try:
        # First try HTTP connection
        client = QdrantClient(
            url=os.getenv("QDRANT_DB_URL", "http://localhost:6333"),
            timeout=5  # Add timeout
        )
        # Test connection
        client.get_collections()
        return client
    except Exception:
        # Fallback to in-memory if remote fails
        print("Falling back to in-memory database")
        return QdrantClient(":memory:")


collection_name = os.getenv("QDRANT_DB_NAME", "image_collection")

image_model_name = "Qdrant/clip-ViT-B-32-vision"
image_model = ImageEmbedding(model_name=image_model_name)

text_model_name = "Qdrant/clip-ViT-B-32-text"
text_model = TextEmbedding(model_name=text_model_name)

payload = "image_path"
