import argparse
import os
import numpy as np
import soundfile as sf
from utils import calculate_metrics

def main(input_dir, output_dir):
    """
    Main function to process each audio file in the input directory, compress and decompress it,
    and then evaluate the compression performance by calculating various metrics.
    
    Parameters:
    input_dir (str): Directory containing the input audio files.
    output_dir (str): Directory to save the decompressed audio files.
    """
    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            # Construct full file paths for the input and output files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".wav", "_decoded.wav"))

            # Encode the input audio file to a compressed format (e.g., .bin)
            os.system(f"./encode {input_path} {output_path.replace('_decoded.wav', '.bin')}")
            # Decode the compressed file back to an audio format (e.g., .wav)
            os.system(f"./decode {output_path.replace('_decoded.wav', '.bin')} {output_path}")

            # Read the original and decoded audio files
            original, sr = sf.read(input_path)
            decoded, _ = sf.read(output_path)

            # Calculate compression metrics
            compression_ratio, snr, psnr, mse, ssim = calculate_metrics(original, decoded)
            
            # Print the results
            print(f"File: {filename}")
            print(f"Compression Ratio: {compression_ratio}")
            print(f"SNR: {snr} dB")
            print(f"PSNR: {psnr} dB")
            print(f"MSE: {mse}")
            print(f"SSIM: {ssim}")

if __name__ == "__main__":
    # Argument parser to get input and output directory paths from the command line
    parser = argparse.ArgumentParser(description="Evaluate the compression performance.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the decoded audio files.")
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args.input_dir, args.output_dir)
