#!/bin/bash
# LLM Interrogator - One command start

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install Python dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Build frontend if not built
if [ ! -f "static/dist/index.html" ]; then
    echo "Building frontend..."
    cd frontend && npm run build && cd ..
fi

echo ""
echo "============================================================"
echo "  LLM Interrogator"
echo "============================================================"
echo "  Open http://localhost:5001 in your browser"
echo "============================================================"
echo ""

# Start Flask
python app.py
