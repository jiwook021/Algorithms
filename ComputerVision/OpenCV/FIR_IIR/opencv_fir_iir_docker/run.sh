#!/bin/bash

# Allow all access to X11 server
xhost +local:docker

# Run application with Docker Compose
docker-compose up --build

# Restore X11 settings after execution
xhost -local:docker
