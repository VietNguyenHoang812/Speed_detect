from fastapi import FastAPI, Body

from src.pipeline import pipeline


app = FastAPI()
@app.get("/")
async def hello():
    response = {
        "message": "For whom the princess weeps",
    }
    return response

@app.post("/inference")
async def inference(image_url: str = Body(...), reserve: int = Body(...)):
    response = pipeline(image_url)
    return response
