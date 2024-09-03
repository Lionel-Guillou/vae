# import general libraries
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# import modules from package
from backend.model import Fashion

app = FastAPI()

RESULTS_DIR = "/Users/lionelguillou/Documents/vae/results"

# Define a Pydantic model for the request body
class RequestGenerate(BaseModel):
    idx: int

@app.post("/generate")
async def generate_request(item: RequestGenerate):
    # retrieve index of object to generate
    idx = item.idx

    # generate object
    model = Fashion()
    array = model(idx, visualize = False)

    # transpose array and get rid of last dimension
    array = array.transpose(1, 2, 0).squeeze()

    # rescale array from 0-1 to 0-255
    array = (array * 255).astype(np.uint8)

    # Convert the NumPy array to a grayscale PIL Image
    image = Image.fromarray(array, mode = "L")

    # save PIL Image in folder
    image.save(os.path.join(RESULTS_DIR, f"res_{idx}.jpg"))

    return "image generated and saved"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
