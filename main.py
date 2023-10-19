import time

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
    start_time = time.time()
    result = pipeline(image_url)
    processed_time = time.time() - start_time
    response = {
        "message": result,
        "processed_time": processed_time
    }
    return response
