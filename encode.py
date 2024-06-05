import argparse
import os
import torch
from hybrid_model import HybridModel
from utils import segment_audio, compress_audio, entropy_encode, file_checksum

def main(args):
    # Load the trained model
    hybrid_model = HybridModel(args.segment_size, args.latent_dim_vqvae, args.latent_dim_cae, args.num_embeddings, args.embedding_dim)
    hybrid_model.load_state_dict(torch.load(args.model_path))
    hybrid_model.eval()

    # Calculate the checksum of the input file
    input_checksum = file_checksum(args.input_file)
    print(f"Input file checksum: {input_checksum}")

    # Segment the audio file
    segments, _ = segment_audio(args.input_file, args.segment_size)

    # Compress the audio segments
    compressed_audio = compress_audio(hybrid_model, segments)

    # Apply entropy encoding
    compressed_audio_encoded = [entropy_encode(indices) for indices in compressed_audio]

    # Save the compressed audio to a file
    with open(args.output_file, 'wb') as f:
        for encoded_indices in compressed_audio_encoded:
            f.write(encoded_indices)

    print(f"Compressed audio saved to: {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Compression Encoder')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the compressed audio file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--segment_size', type=int, default=8000, help='Size of audio segments')
    parser.add_argument('--latent_dim_vqvae', type=int, default=8, help='Latent dimension of VQVAE')
    parser.add_argument('--latent_dim_cae', type=int, default=16, help='Latent dimension of CAE')
    parser.add_argument('--num_embeddings', type=int, default=128, help='Number of embeddings in VQVAE')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Dimension of embeddings in VQVAE')
    args = parser.parse_args()
    main(args)
