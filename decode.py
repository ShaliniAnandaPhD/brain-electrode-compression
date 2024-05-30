import os
import numpy as np
import soundfile as sf
import torch
from model import VQVAE, entropy_decode, decompress_audio

def main(input_file, output_file):
    # Load the trained model
    model = VQVAE(segment_size=16000, latent_dim=2, num_embeddings=128, embedding_dim=2)
    model.load_state_dict(torch.load('vqvae_audio_compression.pth'))
    model.eval()

    # Read the encoded data from the input file
    with open(input_file, 'rb') as f:
        entropy_encoded_audio = []
        while chunk := f.read():
            entropy_encoded_audio.append(chunk)

    # Decode the audio data
    decompressed_audio = decompress_audio(model, entropy_encoded_audio, segment_size=16000)

    # Write the decoded audio to the output file
    sf.write(output_file, decompressed_audio, samplerate=16000)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Decode audio file using VQ-VAE')
    parser.add_argument('input_file', help='Path to the input encoded file')
    parser.add_argument('output_file', help='Path to the output decoded audio file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
