# Brain Electrode Compression

This repository contains a compression algorithm for brain electrode recordings, specifically designed for the N1 implant. The algorithm aims to achieve high compression ratios while maintaining low latency and power consumption, enabling efficient wireless transmission of electrode data.

## Overview

- **Data Generation**: The N1 implant generates approximately 200Mbps of electrode data (1024 electrodes @ 20kHz, 10b resolution) during video game play.
- **Transmission Limitation**: The implant can only transmit around 1Mbps wirelessly.
- **Compression Need**: Reduce data size by more than 200 times while operating in real-time (< 1ms) and consuming low power (< 10mW, including radio).

## Approach

### Vector Quantized Variational Autoencoder (VQ-VAE) Architecture

1. **Preprocessing**: 
   - Segment electrode recordings into smaller chunks (e.g., 16,000 samples) for efficient processing.

2. **VQ-VAE Model**:
   - **Encoder**: Maps the input data to a latent space using convolutional and fully connected layers.
   - **Quantization Layer**: Uses a learned codebook to quantize the latent representations.
   - **Decoder**: Reconstructs the original data from the quantized latent representations.

3. **Training**:
   - The VQ-VAE model is trained using a combination of reconstruction loss and commitment loss to minimize the difference between the original and reconstructed data while encouraging efficient use of the codebook.

4. **Compression**:
   - The trained VQ-VAE model encodes the electrode recordings.
   - The encoder maps input segments to the latent space.
   - The quantization layer assigns each latent vector to the nearest codebook entry.
   - The indices of the assigned codebook entries are entropy-encoded to reduce the data size further.

5. **Decompression**:
   - The entropy-encoded indices are decoded to retrieve the corresponding codebook entries.
   - The decoder reconstructs the original electrode recordings from the quantized latent representations.

6. **Evaluation**:
   - The compression algorithm is evaluated based on the achieved compression ratio, encoding/decoding times, and the entropy of the compressed data.
   - Results are compared to a baseline compression method (e.g., zip) to assess the effectiveness of the VQ-VAE approach.

## Results

| File                                        | Compression Ratio | Original Size (bytes) | Compressed Size (bytes) | Decompressed Size (bytes) | Encoding Time (seconds) | Decoding Time (seconds) | Original Entropy (bits/byte) | Decompressed Entropy (bits/byte) |
|---------------------------------------------|-------------------|-----------------------|-------------------------|---------------------------|--------------------------|--------------------------|----------------------------------|-----------------------------------|
| 49c14ec7-9402-4d5a-898c-e133fe1719a0.wav    | 688.30            | 197,542               | 287                     | 448,000                   | 0.0150                   | 0.0190                   | 16.70                            | 16.63                             |
| 6e91dd23-2f15-48e2-b5d1-1d72ee8d1f3c.wav    | 689.53            | 197,896               | 287                     | 448,000                   | 0.0154                   | 0.0190                   | 16.83                            | 16.64                             |
| 9b06f0d3-83f7-4fd0-9f03-34e0c10560a4.wav    | 687.94            | 197,440               | 287                     | 448,000                   | 0.0185                   | 0.0223                   | 16.64                            | 16.63                             |
| 0aefe960-43fd-41cc-97c8-bf9d2d64efd3.wav    | 688.07            | 197,476               | 287                     | 448,000                   | 0.0158                   | 0.0194                   | 16.89                            | 16.63                             |
| 2985450f-b117-4a49-adb1-73a7a0118505.wav    | 688.33            | 197,550               | 287                     | 448,000                   | 0.0159                   | 0.0191                   | 16.71                            | 16.63                             |
| 47bcd211-fdd3-4b3d-81bb-72595dac9dc2.wav    | 688.10            | 197,486               | 287                     | 448,000                   | 0.0168                   | 0.0197                   | 16.81                            | 16.63                             |
| d6c94651-3e56-4126-8a25-c76b0742203c.wav    | 688.01            | 197,460               | 287                     | 448,000                   | 0.0155                   | 0.0192                   | 16.89                            | 16.63                             |
| 9d3e4007-4faf-4eb7-9bfd-edb27251734d.wav    | 694.40            | 199,294               | 287                     | 448,000                   | 0.0166                   | 0.0205                   | 16.76                            | 16.63                             |

## Requirements

- Python 3.x
- NumPy
- SoundFile
- PyTorch
- SciPy

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShaliniAnandaPhD/brain-electrode-compression.git
   cd brain-electrode-compression
   ```

2. **Install dependencies and train the VQ-VAE model**:
   ```bash
   ./build.sh
   ```

3. **Encode audio files**:
   ```bash
   ./encode.py path/to/input/audio.wav path/to/output/encoded.bin
   ```

4. **Decode audio files**:
   ```bash
   ./decode.py path/to/input/encoded.bin path/to/output/decoded.wav
   ```

## Scripts

- **encode.py**: Script for encoding audio files using the trained VQ-VAE model.
- **decode.py**: Script for decoding compressed audio files using the trained VQ-VAE model.
- **train.py**: Script for training the VQ-VAE model on the audio dataset.
- **model.py**: Contains the implementation of the VQ-VAE model and related utility functions.
- **build.sh**: Shell script for setting up the environment, installing dependencies, and training the VQ-VAE model.

## Evaluation

To verify compression is lossless and measure compression ratio:
```bash
./eval.sh
```

## References

- [Vector Quantized Variational Autoencoders](https://arxiv.org/abs/1711.00937) by Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu.
- [VQ-VAE-2: Learning Hierarchical Latent Representations](https://arxiv.org/abs/1906.00446) by Aaron van den Oord, Yazhe Li, and Oriol Vinyals.

## License

This project is licensed under the MIT License.
