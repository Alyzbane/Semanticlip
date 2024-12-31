import os
import uuid

from pathlib import Path
from tqdm.auto import tqdm

from qdrant_client import models
from config import payload, image_model_name, image_model

def preload_default_images(client):
    """Initial loading of images from default directory when application starts.
    Args:
        client: Qdrant client instance
    """
    collection_name = os.getenv("DB_NAME")
    
    try:
        # Skip if collection already exists
        if client.collection_exists(collection_name):
            print(f"Collection {collection_name} already exists, skipping preload")
            return
            
        # Create new collection
        image_embeddings_size = image_model._get_model_description(image_model_name)["dim"]
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
        
        # Load images from default directory
        process_images(client, os.getenv("DB_FOLDER"))
        
    except Exception as e:
        print(f"Error in preload: {e}")

def process_images(client, folder_path):
    """Process and store image embeddings from a specified folder.
    Skips images that are already in the database based on their file paths.
    """
    try:
        folder_path = Path(folder_path).as_posix() # Use a standard posix folder path
        # Get existing image paths from database
        existing_paths = set()
        # Returns all points in a page-by-page manner. By default, all resulting points are sorted by {id}.
        scroll_response = client.scroll( 
            collection_name=os.getenv("DB_NAME"),
            with_payload=True,
            limit=10000  # Adjust based on your needs
        )
        for point in scroll_response[0]:
            if payload in point.payload:
                existing_paths.add(point.payload[payload])

        extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_paths = []
        
        # Collect new image paths
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(extensions):
                    full_path = os.path.join(root, file)
                    if full_path not in existing_paths:
                        image_paths.append(full_path)

        if not image_paths:
            return f"No new images found in {folder_path}"

        # Process only new images
        points = []
        for image_path in tqdm(image_paths):
            try:
                image_embedding = image_model.embed([image_path])
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"image": list(image_embedding)[0]},
                        payload={payload: image_path},
                    )
                )
                print(f"Embedded and uploaded: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Store new embeddings
        if points:
            client.upsert(collection_name=os.getenv("DB_NAME"), points=points)
            return f"Successfully processed {len(points)} new images from {folder_path}"
        
        return f"No new images were processed from {folder_path}"

    except Exception as e:
        return f"Error processing images: {e}"
