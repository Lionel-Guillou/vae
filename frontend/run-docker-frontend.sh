#!/bin/bash

# Define the image and container name
IMAGE_NAME="frontend"
CONTAINER_NAME="frontend_container"

# Set optional environment variables and port mappings
PORT="3000:80"   # Map port 8000 of the host to port 8000 in the container

# Run the container
echo "Starting a new container from $IMAGE_NAME..."
docker run -d --name $CONTAINER_NAME -p $PORT $IMAGE_NAME