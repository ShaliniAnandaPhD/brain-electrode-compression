import argparse
import os
import torch
import soundfile as sf
from hybrid_model import HybridModel
from utils import decompress_audio, post_process_audio, entropy_decode, file_checksum

def main(args):
    # Load the trained model
    hybrid_model = HybridModel(args.segment_size, args.latent_dim_vqvae, args.latent_dim_cae, args.num_embeddings, args.embedding_dim)
    hybrid_model.load_state_dict(torch.load(args.model_path))
    hybrid_model.eval()

    # Read the compressed audio file
    with open(args.input_file, 'rb') as f:
        compressed_audio = f.read()

    # Split the compressed audio into separate encoded segments
    compressed_audio_encoded = [compressed_audio[i:i+args.segment_size] for i in range(0, len(compressed_audio), args.segment_size)]

    # Decompress the audio
    decompressed_audio = decompress_audio(hybrid_model, compressed_audio_encoded, args.segment_size * len(compressed_audio_encoded))

    # Post-process the decompressed audio
    decompressed_audio = post_process_audio(decompressed_audio, args.samplerate)

    # Save the decompressed audio to a file
    sf.write(args.output_file, decompressed_audio, args.samplerate)

    print(f"Decompressed audio saved to: {args.output_file}")

    # Calculate the checksum of the output file
    output_checksum = file_checksum(args.output_file)
    print(f"Output file checksum: {output_checksum}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Compression Decoder')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the compressed audio file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the decompressed audio file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--segment_size', type=int, default=8000, help='Size of audio segments')
    parser.add_argument('--latent_dim_vqvae', type=int, default=8, help='Latent dimension of VQVAE')
    parser.add_argument('--latent_dim_cae', type=int, default=16, help='Latent dimension of CAE')
    parser.add_argument('--num_embeddings', type=int, default=128, help='Number of embeddings in VQVAE')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Dimension of embeddings in VQVAE')
    parser.add_argument('--samplerate', type=int, default=16000, help='Samplerate of the audio')
    args = parser.parse_args()
    main(args)
