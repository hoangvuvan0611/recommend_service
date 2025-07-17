#!/bin/bash

APP_NAME="flask-recommender"
NETWORK_NAME="agri-network"

# Tạo network nếu chưa có
docker network inspect $NETWORK_NAME >/dev/null 2>&1 || \
    docker network create $NETWORK_NAME

echo "🐳 Build Docker image Flask..."
docker build -t $APP_NAME .

echo "🧹 Dừng và xóa container cũ nếu có..."
docker stop $APP_NAME 2>/dev/null && docker rm $APP_NAME 2>/dev/null

echo "🚀 Run container Flask..."
docker run -d \
  --name $APP_NAME \
  --network $NETWORK_NAME \
  -p 8051:8051 \
  $APP_NAME

echo "✅ Flask chạy tại http://localhost:8051"
