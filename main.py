import os
import torch
from model import HybridModel
from dataset import AudioDataset
from utils import segment_audio, compress_audio, decompress_audio, post_process_audio, entropy_encode, entropy_decode, calculate_metrics, compare_audio

def main():
    """
    Main function to run the audio compression and decompression pipeline.
    """
    data_dir = 'data'
    segment_size = 8000
    latent_dim_vqvae = 8
    latent_dim_cae = 16
    num_embeddings = 128
    embedding_dim = latent_dim_vqvae

    wav_files = [file for file in os.listdir(data_dir) if file.endswith('.wav')]
    first_10_files = wav_files[:10]
    last_10_files = wav_files[-10:]

    segments_list = []
    samplerate_list = []
    for file in first_10_files:
        file_path = os.path.join(data_dir, file)
        segments, samplerate = segment_audio(file_path, segment_size)
        if segments:
            segments_list.extend(segments)
            samplerate_list.append(samplerate)

    dataset = AudioDataset(segments_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    hybrid_model = HybridModel(segment_size, latent_dim_vqvae, latent_dim_cae, num_embeddings, embedding_dim)
    epoch_losses = train_hybrid_model(hybrid_model, dataloader, epochs=100, lr=1e-3)

    compression_ratios = []
    snr_values = []
    psnr_values = []
    mse_values = []
    ssim_values = []

    for file in last_10_files:
        file_path = os.path.join(data_dir, file)
        original_checksum = file_checksum(file_path)
        print(f"Original Checksum for {file}: {original_checksum}")

        segments, samplerate = segment_audio(file_path, segment_size)
        compressed_audio = compress_audio(hybrid_model, segments)
        compressed_audio_encoded = [entropy_encode(indices) for indices in compressed_audio]
        decompressed_audio = decompress_audio(hybrid_model, compressed_audio_encoded, len(segments) * segment_size)
        decompressed_audio = post_process_audio(decompressed_audio, samplerate)

        compression_ratio, original_size, compressed_size, decompressed_size, original_entropy, decompressed_entropy = calculate_metrics(file_path, compressed_audio_encoded, decompressed_audio)
        print(f"Compression ratio for {file}: {compression_ratio:.2f}")
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Decompressed size: {decompressed_size} bytes")
        print(f"Original entropy: {original_entropy:.2f} bits")
        print(f"Decompressed entropy: {decompressed_entropy:.2f} bits")

        is_lossless = compare_audio(file_path, decompressed_audio, samplerate)
        print(f"Lossless compression for {file}: {is_lossless}")

        decompressed_checksum = file_checksum(file_path)
        print(f"Decompressed Checksum for {file}: {decompressed_checksum}")

        original_data, _ = librosa.load(file_path, sr=samplerate, mono=True)
        snr = calculate_snr(original_data, decompressed_audio)
        psnr_value = calculate_psnr(original_data, decompressed_audio)
        mse = calculate_mse(original_data, decompressed_audio)
        ssim_value = calculate_ssim(original_data, decompressed_audio)

        print(f"SNR for {file}: {snr:.2f} dB")
        print(f"PSNR for {file}: {psnr_value:.2f} dB")
        print(f"MSE for {file}: {mse:.6f}")
        print(f"SSIM for {file}: {ssim_value:.4f}")

        compression_ratios.append(compression_ratio)
        snr_values.append(snr)
        psnr_values.append(psnr_value)
        mse_values.append(mse)
        ssim_values.append(ssim_value)
