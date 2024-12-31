import os

import gradio as gr
from PIL import Image

from config import payload, collection_name, image_model, text_model, get_qdrant_client
from embedding import preload_default_images, process_images

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
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("## 🌄 Semantic Image Search")
    gr.Markdown("Created with 😺 Alyzbane [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alyzbane/)")

    
    with gr.Tabs():
        with gr.TabItem("Image Search"):
            gr.Markdown("Upload an image to find similar images")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Upload Image", height=360, scale=1)
                    slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
                    with gr.Row():
                        clear_button = gr.ClearButton()
                        search_button = gr.Button("Search", variant="primary")

                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Search Results", 
                        object_fit="scale-down",
                        show_download_button=False,
                        columns=2
                    )
                clear_button.add([image_input, gallery])

            search_button.click(
                fn=search_by_image,
                inputs=[image_input, slider],
                outputs=gallery
            )

        with gr.TabItem("Text Search"):
            gr.Markdown("Describe and search your image")
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(lines=1, placeholder="Enter text query", label="Text Query")
                    slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
                    with gr.Row():
                        clear_button = gr.ClearButton()
                        text_button = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Search Results", 
                        object_fit="scale-down",
                        columns=2
                    )
                
                clear_button.add([gallery, text_input])
                
            text_button.click(
                fn=search_by_text,
                inputs=[text_input, slider],
                outputs=gallery
            )

        with gr.TabItem("Add Images"):
            gr.Interface(
                fn=lambda folder: process_images(client, folder),
                inputs=gr.Textbox(lines=1, placeholder="Enter folder path", label="Folder Path"),
                outputs=gr.Textbox(label="Status"),
                description="Store images from a folder to the search database",
                flagging_mode='never',
            )

if __name__ == "__main__":
    # Initializing the db client in main
    client = get_qdrant_client()
    if client:
        preload_default_images(client) # Using the 'images' folder to extract embeddings and search from it
        demo.launch(server_name="0.0.0.0")
    else:
        print("Failed to initialize Qdrant client")
