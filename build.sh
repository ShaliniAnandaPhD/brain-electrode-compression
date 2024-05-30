#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Ensure model.py is available in the current directory
if [ ! -f model.py ]; then
  echo "model.py not found! Please ensure model.py is in the current directory."
  exit 1
fi

# Train the VQ-VAE model
python train.py

# Make the encode and decode scripts executable
chmod +x encode.py decode.py

echo "Build completed successfully."
