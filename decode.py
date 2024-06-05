import numpy as np
import torch
import soundfile as sf
import bz2
from model import VQVAE

def decompress_audio(model, compressed_audio, segment_size):
    decompressed_audio = []
    model.eval()
    with torch.no_grad():
        for indices in compressed_audio:
            indices = entropy_decode(indices)
            indices_tensor = torch.tensor(indices).unsqueeze(0)
            z_q = model.codebook(indices_tensor)
            segment_recon = model.decode(z_q)
            decompressed_audio.append(segment_recon.squeeze().numpy())
    decompressed_audio = np.concatenate(decompressed_audio)
    return decompressed_audio[:segment_size * len(compressed_audio)]

def entropy_decode(encoded_indices):
    decompressed_indices = bz2.decompress(encoded_indices)
    indices = np.frombuffer(decompressed_indices, dtype=np.int32)
    return indices

segment_size = 16000
latent_dim = 64
num_embeddings = 512
embedding_dim = 64

model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)
model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
model.eval()

def main(input_file, output_file):
    with open(input_file, 'rb') as f:
        compressed_audio_encoded = []
        while chunk := f.read():
            compressed_audio_encoded.append(chunk)

    decompressed_audio = decompress_audio(model, compressed_audio_encoded, segment_size)
    sf.write(output_file, decompressed_audio, 16000)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
