import os
import librosa
import numpy as np
import bz2
import time
import soundfile as sf
from scipy.stats import entropy, gaussian_kde
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def segment_audio(file_path, segment_size):
    """
    Segment the audio file into smaller segments.

    Args:
        file_path (str): The path to the audio file.
        segment_size (int): The size of each segment.

    Returns:
        tuple: A tuple containing the list of audio segments and the sample rate.
    """
    try:
        audio_data, samplerate = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], 0

    if audio_data.ndim != 1:
        print(f"Error: {file_path} has more than one channel.")
        return [], samplerate

    segments = [audio_data[i:i + segment_size] for i in range(0, len(audio_data), segment_size)]
    for i in range(len(segments)):
        if len(segments[i]) < segment_size:
            segments[i] = np.pad(segments[i], (0, segment_size - len(segments[i])), 'constant')
    print(f"Segmented {file_path} into {len(segments)} segments of size {segment_size}")
    return segments, samplerate

def post_process_audio(audio, samplerate, target_samplerate=16000, silence_threshold=0.01, silence_duration=0.1):
    """
    Post-process the audio by downsampling and trimming silence.

    Args:
        audio (numpy.ndarray): The audio data.
        samplerate (int): The sample rate of the audio.
        target_samplerate (int): The target sample rate for downsampling (default: 16000).
        silence_threshold (float): The threshold for silence detection (default: 0.01).
        silence_duration (float): The duration of silence to trim (default: 0.1).

    Returns:
        numpy.ndarray: The post-processed audio data.
    """
    audio = librosa.resample(audio, orig_sr=samplerate, target_sr=target_samplerate)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20, frame_length=int(target_samplerate * silence_duration), hop_length=int(target_samplerate * silence_duration))
    return audio_trimmed

def entropy_encode(indices):
    """
    Apply entropy coding to the quantized vectors.

    Args:
        indices (numpy.ndarray): The quantized vector indices.

    Returns:
        bytes: The entropy-encoded indices.
    """
    indices_bytes = indices.tobytes()
    compressed_indices = bz2.compress(indices_bytes)
    return compressed_indices

def entropy_decode(encoded_indices):
    """
    Decode the entropy-encoded vectors.

    Args:
        encoded_indices (bytes): The entropy-encoded indices.

    Returns:
        numpy.ndarray: The decoded indices.
    """
    decompressed_indices = bz2.decompress(encoded_indices)
    indices = np.frombuffer(decompressed_indices, dtype=np.int32)
    return indices

def calculate_metrics(original_file, compressed_audio, decompressed_audio):
    """
    Calculate compression ratio, decompressed size, and entropy.

    Args:
        original_file (str): The path to the original audio file.
        compressed_audio (list): The compressed audio data.
        decompressed_audio (numpy.ndarray): The decompressed audio data.

    Returns:
        tuple: A tuple containing the compression ratio, original size, compressed size, decompressed size, original entropy, and decompressed entropy.
    """
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
    """
    Compare the original and decompressed audio data.

    Args:
        original_file (str): The path to the original audio file.
        decompressed_audio (numpy.ndarray): The decompressed audio data.
        samplerate (int): The sample rate of the audio.

    Returns:
        bool: True if the audio is lossless, False otherwise.
    """
    try:
        original_data, _ = librosa.load(original_file, sr=samplerate, mono=True)
    except Exception as e:
        print(f"Error reading file {original_file}: {e}")
        return False

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

def file_checksum(file_path):
    """
    Calculate the checksum of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The checksum of the file.
    """
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_snr(original, decompressed):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the original and decompressed audio.

    Args:
        original (numpy.ndarray): The original audio data.
        decompressed (numpy.ndarray): The decompressed audio data.

    Returns:
        float: The SNR value in decibels.
    """
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - decompressed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_mse(original, decompressed):
    """
    Calculate the Mean Squared Error (MSE) between the original and decompressed audio.

    Args:
        original (numpy.ndarray): The original audio data.
        decompressed (numpy.ndarray): The decompressed audio data.

    Returns:
        float: The MSE value.
    """
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    return np.mean((original - decompressed) ** 2)

def calculate_psnr(original, decompressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and decompressed audio.

    Args:
        original (numpy.ndarray): The original audio data.
        decompressed (numpy.ndarray): The decompressed audio data.

    Returns:
        float: The PSNR value in decibels.
    """
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    mse_value = calculate_mse(original, decompressed)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def calculate_ssim(original, decompressed):
    """
    Calculate the Structural Similarity Index (SSIM) between the original and decompressed audio.

    Args:
        original (numpy.ndarray): The original audio data.
        decompressed (numpy.ndarray): The decompressed audio data.

    Returns:
        float: The SSIM value.
    """
    min_length = min(len(original), len(decompressed))
    original = original[:min_length]
    decompressed = decompressed[:min_length]

    return ssim(original, decompressed)
