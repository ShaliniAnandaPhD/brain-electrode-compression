import numpy as np
import torch
import soundfile as sf
import bz2
from model import VQVAE

# Define decompress_audio and entropy_decode functions
def decompress_audio(model, compressed_audio, segment_size):
    decompressed_audio = []
    model.eval()
    with torch.no_grad():
        for indices in compressed_audio:
            indices = entropy_decode(indices)
            indices_tensor = torch.tensor(indices).unsqueeze(0)
            z_q = model.embedding(indices_tensor)
            recon_segment = model.decode(z_q)
            decompressed_audio.append(recon_segment.squeeze().numpy())
    decompressed_audio = np.concatenate(decompressed_audio)
    return decompressed_audio[:segment_size * len(compressed_audio)]  # Truncate to original length

def entropy_decode(encoded_indices):
    decompressed_indices = bz2.decompress(encoded_indices)
    indices = np.frombuffer(decompressed_indices, dtype=np.int64)
    return indices

data_dir = 'data'
segment_size = 16000
latent_dim = 2
num_embeddings = 128
embedding_dim = latent_dim

# Load the trained model
model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)
model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
model.eval()

# Decode function
def decode(input_file, output_file):
    with open(input_file, 'rb') as f:
        entropy_encoded_audio = []
        while chunk := f.read():
            entropy_encoded_audio.append(chunk)
    
    decompressed_audio = decompress_audio(model, entropy_encoded_audio, segment_size)
    
    sf.write(output_file, decompressed_audio, 16000)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    decode(input_file, output_file)

