from fastapi import FastAPI, Body



app = FastAPI()

@app.post("/")
async def inference(image_urls: list = Body(...)):
    return {"message": "Hello World"}