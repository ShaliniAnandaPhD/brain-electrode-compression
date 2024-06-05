# main.py
import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from model import VQVAE, AudioDataset, train_vqvae, compress_audio, decompress_audio, entropy_encode, entropy_decode
from utils import calculate_metrics, compare_audio, calculate_snr, calculate_psnr, calculate_mse, calculate_ssim

# Set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Set the data directory and segment size
data_dir = 'data'
segment_size = 16000

# Set the model hyperparameters
latent_dim = 64
num_embeddings = 512
embedding_dim = 64

# Get the list of .wav files in the directory
wav_files = [file for file in os.listdir(data_dir) if file.endswith('.wav')]

# Segment the audio files and create the dataset
segments_list = []
samplerate_list = []
for file in wav_files:
    file_path = os.path.join(data_dir, file)
    segments, samplerate = segment_audio(file_path, segment_size)
    segments_list.extend(segments)
    samplerate_list.append(samplerate)

dataset = AudioDataset(segments_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the VQ-VAE model
model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)

# Train the VQ-VAE model
train_vqvae(model, dataloader, epochs=100, learning_rate=1e-3)

# Compress and decompress the audio files
for file in wav_files:
    file_path = os.path.join(data_dir, file)
    segments, samplerate = segment_audio(file_path, segment_size)
    
    # Compress the audio segments
    compressed_audio = compress_audio(model, segments)
    compressed_audio_encoded = [entropy_encode(indices) for indices in compressed_audio]
    
    # Decompress the audio segments
    decompressed_audio = decompress_audio(model, compressed_audio_encoded, segment_size)
    
    # Calculate compression metrics
    compression_ratio, original_size, compressed_size, decompressed_size, original_entropy, decompressed_entropy = calculate_metrics(file_path, compressed_audio_encoded, decompressed_audio)
    print(f"Compression ratio for {file}: {compression_ratio:.2f}")
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Decompressed size: {decompressed_size} bytes")
    print(f"Original entropy: {original_entropy:.2f} bits")
    print(f"Decompressed entropy: {decompressed_entropy:.2f} bits")
    
    # Compare the original and decompressed audio
    is_lossless = compare_audio(file_path, decompressed_audio, samplerate)
    print(f"Lossless compression for {file}: {is_lossless}")
    
    # Calculate additional metrics
    original_data, _ = sf.read(file_path)
    snr = calculate_snr(original_data, decompressed_audio)
    psnr_value = calculate_psnr(original_data, decompressed_audio)
    mse = calculate_mse(original_data, decompressed_audio)
    ssim_value = calculate_ssim(original_data, decompressed_audio)
    
    print(f"SNR for {file}: {snr:.2f} dB")
    print(f"PSNR for {file}: {psnr_value:.2f} dB")
    print(f"MSE for {file}: {mse:.6f}")
    print(f"SSIM for {file}: {ssim_value:.4f}")
    
    print("--------------------")
