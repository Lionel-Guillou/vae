#!/bin/bash

# Define the image and container name
IMAGE_NAME="backend"
CONTAINER_NAME="backend_container"

# Set optional environment variables and port mappings
PORT="8000:8000"   # Map port 8000 of the host to port 8000 in the container

# Set environment variables for directories where results are saved
RESULTS_DIR_LOCAL="/Users/lionelguillou/Documents/vae/results"
RESULTS_DIR_DOCKER="/docker_results"

# Run the container
echo "Starting a new container from $IMAGE_NAME..."
docker run -d --name $CONTAINER_NAME -p $PORT -v $RESULTS_DIR_LOCAL:$RESULTS_DIR_DOCKER $IMAGE_NAME