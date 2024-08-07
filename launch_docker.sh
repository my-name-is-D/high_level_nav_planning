#!/bin/bash

xhost +local:root


# Path to the folder on your desktop
DESKTOP_RESULTS_FOLDER="$(pwd)/results"

# Run the Docker container with GUI support (works on CPU here, not NVIDIA)
docker run -it -d --rm \
    --name high_nav \
    -e DISPLAY=$DISPLAY \
    -w /higher_level_nav \
    -v ${DESKTOP_RESULTS_FOLDER}:/higher_level_nav/results \
    -v /tmp/.X11-unix:/tmp/.X11-unix high_nav:latest
    

docker exec -it high_nav bash