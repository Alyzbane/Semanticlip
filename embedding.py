import os
import uuid

from tqdm.auto import tqdm

from qdrant_client import models
from config import payload, image_model_name, image_model

# Preload database with images from directory
def load_images(client, custom_folder=None):
    """Process and store image embeddings in Qdrant database.
    Args:
        client: Qdrant client instance
        custom_folder: Optional user-provided folder path
    Returns:
        str: Status message for UI feedback when custom_folder is used
    """
    collection_name = os.getenv("DB_NAME")
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

        # Use custom folder if provided, else use default from env
        image_dir = custom_folder if custom_folder else os.getenv("DB_FOLDER")
        extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_paths = []

        # Recursively collect all image files from the folder
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(extensions):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            message = f"No supported images found in {image_dir}"
            return message if custom_folder else print(message)

        points = []
        for image_path in tqdm(image_paths):
            try:
                # Generate CLIP embedding for image
                image_embedding = image_model.embed([image_path])
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"image": list(image_embedding)[0]}, # Store the image embeddings
                        payload={payload: image_path}, # Store original image path
                    )
                )
                print(f"Embedded and uploaded: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Store all embeddings in database
        if points:
            client.upsert(collection_name=collection_name, points=points)
            message = f"Successfully processed {len(points)} images from {image_dir}"
        else:
            message = f"Error: No images were successfully processed from {image_dir}"
        
        return message if custom_folder else print(message)

    except Exception as e:
        message = f"Error initializing collection: {e}"
        return message if custom_folder else print(message)
