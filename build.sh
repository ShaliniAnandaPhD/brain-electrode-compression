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
echo "#!/usr/bin/env python" > encode
cat <<EOF >> encode
import sys
from encode import main as encode_main

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    encode_main(input_file, output_file)
EOF
chmod +x encode

# Create the decode executable
echo "#!/usr/bin/env python" > decode
cat <<EOF >> decode
import sys
from decode import main as decode_main

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    decode_main(input_file, output_file)
EOF
chmod +x decode

echo "Build completed successfully."
