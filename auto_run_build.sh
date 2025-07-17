#!/bin/bash

APP_NAME="flask-recommender"

# Build Docker image
docker build -t $APP_NAME .

# Stop & remove container nếu đã chạy trước đó
docker stop $APP_NAME 2>/dev/null && docker rm $APP_NAME 2>/dev/null

# Run Docker container
docker run -d \
  --name $APP_NAME \
  -p 8051:8051 \
  $APP_NAME
