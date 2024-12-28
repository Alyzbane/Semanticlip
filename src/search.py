from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name =  collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")
    
    def serach(self, text: str):
        # Conversion Text query -> vector
        vector = self.model.encode(text).tolist()

        # Use 'vector' for searching the closest vectors in collections
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,
            limit=5, # Total returned
        ).points

        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]

        return payloads


