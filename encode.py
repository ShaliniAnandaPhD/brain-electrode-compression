import os
import numpy as np
import soundfile as sf
import torch
from model import VQVAE, segment_audio, compress_audio, entropy_encode

def main(input_file, output_file):
    # Load the trained model
    model = VQVAE(segment_size=16000, latent_dim=2, num_embeddings=128, embedding_dim=2)
    model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
    model.eval()

    # Encode the audio file
    segments, _ = segment_audio(input_file, segment_size=16000)
    compressed_audio = compress_audio(model, segments)
    entropy_encoded_audio = [entropy_encode(c) for c in compressed_audio]

    # Write the encoded data to the output file
    with open(output_file, 'wb') as f:
        for segment in entropy_encoded_audio:
            f.write(segment)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio file using VQ-VAE')
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('output_file', help='Path to the output encoded file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
