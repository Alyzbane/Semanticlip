import os

import gradio as gr
from PIL import Image

from config import payload, collection_name, image_model, text_model, get_qdrant_client
from embedding import preload_images

# ========================== Building an UI with gradio  ==========================
def search_by_image(image_path):
    query_image_embedding = image_model.embed([image_path])
    results = client.search(
        collection_name=collection_name,
        query_vector=("image", list(query_image_embedding)[0]),
        with_payload=[payload],
        limit=10
    )
    output_images = []
    for result in results:
        print(result.payload)
        if payload in result.payload:
            output_images.append(Image.open(result.payload[payload]))
        else:
            print(f"Warning: {payload} missing in payload for result: {result}")

    return output_images

def search_by_text(text_query):
    text_query_embedding = text_model.embed([text_query])
    results = client.search(
        collection_name=collection_name,
        query_vector=("image", list(text_query_embedding)[0]),
        with_payload=[payload],
        limit=10
    )

    output_images = []
    for result in results:
        if payload in result.payload:
            output_images.append(Image.open(result.payload[payload]))
        else:
            print(f"Warning: {payload} missing in payload for result: {result}")

    return output_images

with gr.Blocks() as demo:
    gr.Markdown("## Qdrant Image Search")

    with gr.Tabs():
        with gr.TabItem("Image Search"):
            image_input = gr.Image(type="filepath", label="Upload Image")
            image_button = gr.Button("Search by Image")
            image_output = gr.Gallery(label="Search Results", object_fit="scale-down")
            image_button.click(search_by_image, inputs=image_input, outputs=image_output)

        with gr.TabItem("Text Search"):
            text_input = gr.Textbox(lines=1, placeholder="Enter text query", label="Text Query")
            text_button = gr.Button("Search by Text")
            text_output = gr.Gallery(label="Search Results", object_fit="scale-down")
            text_button.click(search_by_text, inputs=text_input, outputs=text_output)

if __name__ == "__main__":
    client = get_qdrant_client()
    if client:
        preload_images(client)
        demo.launch(server_name="0.0.0.0")
    else:
        print("Failed to initialize Qdrant client")
