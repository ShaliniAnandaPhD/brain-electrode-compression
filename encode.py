import os
import numpy as np
import soundfile as sf
import torch
import bz2
from model import VQVAE, segment_audio

def compress_audio(model, segments):
    compressed_audio = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment_tensor = torch.tensor(segment).float().unsqueeze(0)
            z = model.encode(segment_tensor)
            _, indices = model.vector_quantize(z)
            compressed_audio.append(indices.squeeze().numpy())
    compressed_audio = np.array(compressed_audio, dtype=np.int32)
    return compressed_audio

def entropy_encode(indices):
    indices_bytes = indices.tobytes()
    compressed_indices = bz2.compress(indices_bytes)
    return compressed_indices

segment_size = 16000
latent_dim = 64
num_embeddings = 512
embedding_dim = 64

model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)
model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
model.eval()

def main(input_file, output_file):
    segments, samplerate = segment_audio(input_file, segment_size)
    compressed_audio = compress_audio(model, segments)
    compressed_audio_encoded = [entropy_encode(indices) for indices in compressed_audio]

    with open(output_file, 'wb') as f:
        for encoded_segment in compressed_audio_encoded:
            f.write(encoded_segment)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
