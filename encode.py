import os
import numpy as np
import soundfile as sf
import torch
import bz2
from model import VQVAE, segment_audio

# Define compress_audio and entropy_encode functions
def compress_audio(model, segments):
    compressed_audio = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment_tensor = torch.tensor(segment).float().unsqueeze(0).view(1, -1)
            z = model.encode(segment_tensor)
            z_q, indices = model.quantize(z)
            compressed_audio.append(indices.squeeze().numpy())
    compressed_audio = np.array(compressed_audio)
    return compressed_audio

def entropy_encode(indices):
    indices_bytes = indices.tobytes()
    compressed_indices = bz2.compress(indices_bytes)
    return compressed_indices

data_dir = 'data'
segment_size = 16000
latent_dim = 2
num_embeddings = 128
embedding_dim = latent_dim

# Load the trained model
model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)
model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
model.eval()

# Encode function
def encode(input_file, output_file):
    segments, samplerate = segment_audio(input_file, segment_size)
    compressed_audio = compress_audio(model, segments)
    entropy_encoded_audio = [entropy_encode(c) for c in compressed_audio]
    
    with open(output_file, 'wb') as f:
        for segment in entropy_encoded_audio:
            f.write(segment)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    encode(input_file, output_file)
