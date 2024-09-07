# import general libraries
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from dynaconf import Dynaconf
from contextlib import asynccontextmanager

# import modules from package
from backend.model import Fashion

# Get the directory of the current module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the grandparent directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Load configuration from the config/settings.toml file
config = Dynaconf(settings_files=[f'{PARENT_DIR}/config/settings.toml'],
                  environments=True)

# Initialize variable to hold model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event - This is executed when the application starts
    print("Starting up the app...")
    
    # Perform any startup tasks, like loading models, setting up databases, etc.
    global model
    model = Fashion()
    model.eval()
    print("Model successfully loaded!")

    # You can also yield shared resources here if needed.
    
    yield  # Yield control to allow the app to start serving requests

    # Shutdown event - This is executed when the application is shutting down
    print("Shutting down the app...")
    # Perform any cleanup tasks, like closing database connections, etc.

# Initialize the FastAPI app with the lifespan handler
app = FastAPI(lifespan=lifespan)

# Define a Pydantic model for the request body
class RequestGenerate(BaseModel):
    idx: int

@app.post("/generate")
async def generate_request(item: RequestGenerate):

    global model
    if model is None:
        return {"error": "Model not loaded"}

    # retrieve index of object to generate
    idx = item.idx

    # generate object
    array = model(idx, visualize = False)

    # transpose array and get rid of last dimension
    array = array.transpose(1, 2, 0).squeeze()

    # rescale array from 0-1 to 0-255
    array = (array * 255).astype(np.uint8)

    # Convert the NumPy array to a grayscale PIL Image
    image = Image.fromarray(array, mode = "L")

    # save PIL Image in folder
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    image.save(os.path.join(config.RESULTS_DIR, f"res_{idx}.jpg"))

    return "image generated and saved"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
