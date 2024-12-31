import os

import gradio as gr
from PIL import Image

from config import payload, collection_name, image_model, text_model, get_qdrant_client
from embedding import load_images

def search_by_image(image_path, limit=5):
    query_image_embedding = image_model.embed([image_path])
    results = client.search(
        collection_name=collection_name,
        query_vector=("image", list(query_image_embedding)[0]),
        with_payload=[payload],
        limit=limit
    )
    output_images = []
    for result in results:
        print(result.payload)
        if payload in result.payload:
            output_images.append(Image.open(result.payload[payload]))
        else:
            print(f"Warning: {payload} missing in payload for result: {result}")

    return output_images

def search_by_text(text_query, limit=5):
    text_query_embedding = text_model.embed([text_query])
    results = client.search(
        collection_name=collection_name,
        query_vector=("image", list(text_query_embedding)[0]),
        with_payload=[payload],
        limit=limit
    )

    output_images = []
    for result in results:
        if payload in result.payload:
            output_images.append(Image.open(result.payload[payload]))
        else:
            print(f"Warning: {payload} missing in payload for result: {result}")

    return output_images

# ========================== Building an UI with gradio  ==========================
slider  = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
gallery = gr.Gallery(label="Search Results", object_fit="scale-down")

with gr.Blocks() as demo:
    gr.Markdown("## Semantic Image Search")
        
    with gr.Tabs():
        with gr.TabItem("Image Search"):
            gr.Interface(
                fn=search_by_image,
                inputs=[
                    gr.Image(type="filepath", label="Upload Image", height=360),
                    slider,
                ],
                outputs=gallery,
                description="Upload an image to find similar images",
                flagging_mode='never',
            )

        with gr.TabItem("Text Search"):
            gr.Interface(
                fn=search_by_text,
                inputs=[
                    gr.Textbox(lines=1, placeholder="Enter text query", label="Text Query"),
                    slider,
                ],
                outputs=gallery,
                description="Enter text to find matching images",
                flagging_mode='never',
            )
        with gr.TabItem("Add Images"):
            gr.Interface(
                fn=lambda folder: load_images(client, folder),
                inputs=gr.Textbox(lines=1, placeholder="Enter folder path", label="Folder Path"),
                outputs=gr.Textbox(label="Status"),
                description="Add images from a folder to the search database",
                flagging_mode='never',
            )


if __name__ == "__main__":
    # Initializing the db client in main
    client = get_qdrant_client()
    if client:
        load_images(client) # Using the 'images' folder to extract embeddings and search from it
        demo.launch(server_name="0.0.0.0")
    else:
        print("Failed to initialize Qdrant client")
