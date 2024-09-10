# import external libraries
from fastapi import FastAPI, Request # type: ignore
import uvicorn # type: ignore
from pydantic import BaseModel # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
from dynaconf import Dynaconf # type: ignore
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define a Pydantic model for the request body
class RequestGenerate(BaseModel):
    idx: int

@app.get("/welcome")
async def read_status():
    # Python dictionary, which will be converted to JSON by FastAPI
    status = {
        "message": "Welcome to the Fashion Generation API",
    }
    return status

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

    return {"msg": "image generated and saved"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
