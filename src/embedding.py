import os
import uuid

from tqdm.auto import tqdm

from qdrant_client import models
from config import payload, image_model_name, image_model

# Preload database with images from directory
def preload_images(client):
    collection_name = os.getenv("QDRANT_DB_NAME", "image_collection")
    image_embeddings_size = image_model._get_model_description(image_model_name)["dim"]
    
    try:
        # Create collection if it doesn't exist
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "image": models.VectorParams(
                        size=image_embeddings_size, 
                        distance=models.Distance.COSINE
                    ),
                }
            )
            print(f"Created collection: {collection_name}")

            image_dir = "../images"
            extensions = ('.jpg', '.jpeg', '.png', '.webp')
            image_paths = []

            for entry in tqdm(os.scandir(image_dir), total=len(os.listdir(image_dir))):
                if entry.is_file() and entry.path.endswith(extensions):
                    image_paths.append(entry.path)

            points = []
            for image_path in tqdm(image_paths):
                try:
                    image_embedding = image_model.embed([image_path]) #embedding the image
                    point_id = str(uuid.uuid4()) #unique id for each image
                    points.append(
                        models.PointStruct(
                            id=point_id, # use filename as id
                            vector={"image": list(image_embedding)[0]}, # embedding
                            payload={payload: image_path}, # user-defined payload
                        )
                    )
                    print(f"Embedded and uploaded: {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

            if len(points) != 0:
                client.upsert(collection_name=collection_name, points=points)
            else:
                print(f"Error: Empty collection, cannot be upated ")
    except Exception as e:
        print(f"Error initializing collection: {e}")

