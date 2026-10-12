#!/bin/bash
# Start script for RAG Application

set -e

echo "Starting RAG Application..."

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    docker-compose up --build
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "Using Docker Compose V2..."
    docker compose up --build
else
    echo "Docker Compose not found. Starting manually..."
    
    # Start backend
    echo "Starting backend..."
    cd backend
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r requirements.txt -q
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    echo "Starting frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
    echo "API Docs: http://localhost:8000/docs"
    
    # Wait for processes
    wait $BACKEND_PID $FRONTEND_PID
fi