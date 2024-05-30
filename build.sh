#!/bin/bash

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Train the VQ-VAE model
python train.py

# Make the encode and decode scripts executable
chmod +x encode.py decode.py
