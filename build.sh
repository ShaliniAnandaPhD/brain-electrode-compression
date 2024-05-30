#!/bin/bash

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Copy the model.py file to the current directory
cp /path/to/model.py .

# Train the VQ-VAE model
python train.py

# Make the encode and decode scripts executable
chmod +x encode.py decode.py
