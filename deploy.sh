#!/bin/bash

# Step 1: Get the image name from the argument
if [ -z "$1" ]; then
  echo "Usage: $0 <image_name>"
  exit 1
fi

IMAGE_NAME=$1

# Step 2: Check if there is an existing running container based on the image name
EXISTING_CONTAINER=$(docker ps -q -f ancestor="$IMAGE_NAME")

if [ -n "$EXISTING_CONTAINER" ]; then
  # Step 3: Stop and remove the existing container
  echo "Stopping and removing existing container..."
  docker stop "$EXISTING_CONTAINER" && docker rm "$EXISTING_CONTAINER"
fi

# Step 4: Build the Docker image
echo "Building the Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .

# Step 5: Run the new instance
echo "Running the container..."

docker run -td --rm -e DISPLAY=host.docker.internal:0 -p 8080:8080 -v "/tmp/.X11-unix:/tmp/.X11-unix" -e NUM_EPISODES=1 "$IMAGE_NAME"
