import json
import numpy as np

# Import client library
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Run a docker container with localhost port
# docker run -p 6333:6333 \
#     -v $(pwd)/qdrant_storage:/qdrant/storage \
#     qdrant/qdrant

client = QdrantClient("http://localhost:6333") # 


if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

fd = open("./startups_demo.json")

# iterator over data
payload = map(json.loads, fd)

# laod all vector in memory
# Use Mmap to access data on the fly
vectors = np.load("./startup_vectors.npy")

client.upload_collection(
    collection_name="startups",
    vectors=vectors,
    payload=payload,
    ids=None, # Vector ids assigned automatically
    batch_size=256, # How many vectors will be uploaded in a single result
)

