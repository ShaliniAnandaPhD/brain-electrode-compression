#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Ensure model.py is available by copying it from the GitHub repository
MODEL_URL="https://raw.githubusercontent.com/ShaliniAnandaPhD/brain-electrode-compression/main/model.py"
curl -o model.py $MODEL_URL

# Check if model.py was downloaded successfully
if [ ! -f model.py ]; then
  echo "Failed to download model.py! Please ensure the URL is correct and you have internet access."
  exit 1
fi

# Train the VQ-VAE model
python train.py

# Create the encode executable
cat <<EOF > encode.py
#!/usr/bin/env python
import argparse
from encode import main as encode_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Compression Encoder')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the compressed audio file')
    args = parser.parse_args()
    encode_main(args.input_file, args.output_file)
EOF

# Make the encode script executable
chmod +x encode.py

# Build the encode executable using PyInstaller
pyinstaller --onefile encode.py --name encode

# Create the decode executable
cat <<EOF > decode.py
#!/usr/bin/env python
import argparse
from decode import main as decode_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Compression Decoder')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the compressed audio file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the decompressed audio file')
    args = parser.parse_args()
    decode_main(args.input_file, args.output_file)
EOF

# Make the decode script executable
chmod +x decode.py

# Build the decode executable using PyInstaller
pyinstaller --onefile decode.py --name decode

# Clean up the build files
rm -rf build dist *.spec encode.py decode.py

# Deactivate the virtual environment
deactivate

echo "Build completed successfully."
