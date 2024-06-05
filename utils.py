# utils.py
import os
import numpy as np
import soundfile as sf
from scipy.stats import entropy, gaussian_kde
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def segment_audio(file_path, segment_size):
    audio_data, samplerate = sf.read(file_path)
    segments = [audio_data[i:i + segment_size] for i in range(0, len(audio_data), segment_size)]
    for i in range(len(segments)):
        if len(segments[i]) < segment_size:
            segments[i] = np.pad(segments[i], (0, segment_size - len(segments[i])), 'constant')
    return segments, samplerate

def calculate_metrics(original_file, compressed_audio, decompressed_audio):
    original_size = os.path.getsize(original_file)
    compressed_size = sum(len(c) for c in compressed_audio)
    decompressed_size = decompressed_audio.nbytes
    compression_ratio = original_size / compressed_size

    original_entropy = entropy(np.frombuffer(open(original_file, 'rb').read(), dtype=np.uint8), base=2)

    decompressed_data = decompressed_audio.astype(np.float64)
    decompressed_data += 1e-8
    decompressed_kde = gaussian_kde(decompressed_data)
    decompressed_entropy = entropy(decompressed_kde.evaluate(decompressed_data), base=2)

    return compression_ratio, original_size, compressed_size, decompressed_size, original_entropy, decompressed_entropy

def compare_audio(original_file, decompressed_audio, samplerate):
    original_data, _ = sf.read(original_file)
    original_length = len(original_data)
    decompressed_length = len(decompressed_audio)
    if original_length != decompressed_length:
        min_length = min(original_length, decompressed_length)
        original_data = original_data[:min_length]
        decompressed_audio = decompressed_audio[:min_length]

    original_data = original_data / np.max(np.abs(original_data))
    decompressed_audio = decompressed_audio / np.max(np.abs(decompressed_audio))

    diff = np.abs(original_data - decompressed_audio)
    max_diff = np.max(diff)
    lossless_threshold = 1e-6
    if max_diff <= lossless_threshold:
        return True
    else:
        return False

def calculate_snr(original, decompressed):
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - decompressed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_mse(original, decompressed):
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    return np.mean((original - decompressed) ** 2)

def calculate_psnr(original, decompressed):
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    mse_value = calculate_mse(original, decompressed)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def calculate_ssim(original, decompressed):
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    return ssim(original, decompressed)
