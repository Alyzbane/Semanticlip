from fastapi import FastAPI

# Defined class
from search import NeuralSearcher

app = FastAPI()

# NeuralSearcher instance
neural_searcher = NeuralSearcher(collection_name="startups")

@app.get("/api/search")
def search_startup(q: str):
    return {"resul": neural_searcher.serach(text=q)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)