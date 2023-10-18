from fastapi import FastAPI, Body

from src.pipeline import pipeline

app = FastAPI()

@app.post("/inference")
async def inference(image_url: str = Body(...)):
    response = pipeline(image_url)
    return {"message": response}