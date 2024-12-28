import os
import glob

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from tqdm.cli import tqdm

from fastembed import TextEmbedding, ImageEmbedding

def text_embeddings():
    # Using GPU by default
    model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda" # or 'cpu'
    ) 

    df = pd.read_json(r'./startups_demo.json', lines=True)

    vectors = model.encode(
        [row.alt + ": " + row.description for row in df.itertuples()],
        show_progress_bar=True,
    )

    print(vectors.shape)

    # Saving the vector embeddings
    np.save("startup_vectors.npy", vectors, allow_pickle=False)


def image_embeddings():
    directory = r'..\images'
    extension = '.jpg'

    images = []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.path.endswith(extension):
            images.append(entry.path)
    
    image_model_name = "Qdrant/clip-ViT-B-32-vision" # CLIP image encoder
    image_model = ImageEmbedding(
            model_nam=image_model_name
            providers=["CUDAExecutionProvider"] # FastEmbed uses onnxruntime
        )
    image_embedded_size = image_model._get_model_description(image_model)["dim"] # Dimension of image embeddings
    image_encoded = list(image_model.embed(images)) # Embed images with CLIP encoder
    

if __name__ == "__main__":
    image_embeddings()
