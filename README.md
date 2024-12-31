# SemanticCLIP Image Search

A semantic image search application using CLIP embeddings and Qdrant vector database. Search through your image collection using natural language queries.

## Features

- Natural language image search
- Vector similarity search using CLIP embeddings
- Support for both persistent storage (Docker) and in-memory database
- Supports jpg, jpeg, png, and webp image formats

## Prerequisites

- Python 3.10+
- Docker (optional - for persistent storage)
- 4GB+ RAM
- GPU recommended but not required


## Quick Start

1. Clone the repository:
```
git clone https://github.com/Alyzbane/Semanticlip.git
cd Semanticlip
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
# or
.\venv\Scripts\activate # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```

## Setup Options

### Option 1: Using Docker (Recommended)

1. Start Qdrant container:
```
docker-compose up -d
```
2. Modify .env file as you wish
```
DB_URL=http://localhost:6333
DB_NAME=image_collection
```

## Option 2: In-Memory Database 
No additional setup required - the application will automatically use in-memory storage if Qdrant server is not available.

### Usage
1. Place your images in the images directory:
```
semanticclip/
└── images/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

2. Run the application:
```
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:7860
```

## Troubleshooting
- Database Connection Issues: If the application can't connect to Qdrant, it will automatically fall back to in-memory storage
    - If, docker is running:
    ```powershell
    curl http://localhost:6333 # test the connection

    # Expected Output:
    StatusCode        : 200
    StatusDescription : OK
    ...
    ```

- Memory Errors: Reduce batch size or number of images if encountering memory issues
- Docker Issues: Ensure proper permissions on the storage directory

## Technical Details
View supported models [here.](https://qdrant.github.io/fastembed/examples/Supported_Models/) 

Modify models setup at **config.py**. 
- Vector Database: Qdrant
- Embedding Model: CLIP ViT-B/32
- Image Types: jpg, jpeg, png, webp
- Vector Dimension: 512 
- Distance Metric: Cosine Similarity

## Limitations
- In-memory storage is temporary and will be cleared when the application stops
- Large image collections require significant RAM when using in-memory storage
- Processing speed depends on hardware capabilities