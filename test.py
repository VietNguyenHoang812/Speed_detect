import requests
import json


# defining the api-endpoint
API_ENDPOINT = "http://127.0.0.1:8000/inference"

# data to be sent to api
data = {
    "image_url": "/app/images/test.JPG",
    "reserve": 1
}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, json=data)
print(r.json())
