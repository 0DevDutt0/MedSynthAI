#!/bin/bash

# Script to pull Ollama models and setup embedding client
# This script is idempotent and will start Ollama server if not running

set -e  # Exit on any error

echo "Checking if Ollama server is running..."
if pgrep -f "ollama serve" > /dev/null; then
    echo "Ollama server is already running"
else
    echo "Starting Ollama server..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5  # Give server time to start
fi

echo "Checking if nomic-embed-text model is available..."
if ollama list | grep -q "nomic-embed-text"; then
    echo "nomic-embed-text model already pulled"
else
    echo "Pulling nomic-embed-text model..."
    ollama pull nomic-embed-text
fi

echo "Checking if tinyllama model is available..."
if ollama list | grep -q "tinyllama"; then
    echo "tinyllama model already pulled"
else
    echo "Pulling tinyllama model..."
    ollama pull tinyllama
fi

echo "Ollama setup complete!"