# Brain Electrode Compression

This repository contains a compression algorithm for brain electrode recordings, specifically designed for the N1 implant. The algorithm aims to achieve high compression ratios while maintaining low latency and power consumption, enabling efficient wireless transmission of electrode data.

## Overview

The N1 implant, located in the motor cortex of a non-human primate, generates approximately 200Mbps of electrode data (1024 electrodes @ 20kHz, 10b resolution) during video game play. However, the implant can only transmit around 1Mbps wirelessly. To overcome this limitation, a compression algorithm is necessary to reduce the data size by more than 200 times while operating in real-time (< 1ms) and consuming low power (< 10mW, including radio).

This repository explores a novel approach to compressing brain electrode recordings using a Vector Quantized Variational Autoencoder (VQ-VAE) architecture.

## Approach

The compression algorithm utilizes a VQ-VAE architecture to compress the electrode recordings. The VQ-VAE consists of an encoder, a quantization layer, and a decoder. The encoder maps the input data to a latent space, which is then quantized using a learned codebook. The quantized latent representations are then decoded back to the original data space.

The key components of the compression approach are as follows:

1. **Preprocessing**: The electrode recordings, provided as uncompressed monochannel WAV files, are segmented into smaller chunks of a fixed size (e.g., 16,000 samples) to facilitate efficient processing.

2. **VQ-VAE Architecture**: The VQ-VAE model is designed with a reduced latent space dimension and a smaller codebook size to achieve high compression ratios. The encoder and decoder are implemented using convolutional and fully connected layers, respectively.

3. **Training**: The VQ-VAE model is trained on a subset of the electrode recordings using a combination of reconstruction loss and commitment loss. The training process aims to minimize the difference between the original and reconstructed data while encouraging the model to utilize the learned codebook effectively.

4. **Compression**: During compression, the trained VQ-VAE model is used to encode the electrode recordings. The encoder maps the input segments to the latent space, and the quantization layer assigns each latent vector to the nearest codebook entry. The indices of the assigned codebook entries are then entropy-encoded to further reduce the data size.

5. **Decompression**: To decompress the data, the entropy-encoded indices are first decoded, and the corresponding codebook entries are retrieved. The decoder then reconstructs the original electrode recordings from the quantized latent representations.

6. **Evaluation**: The compression algorithm is evaluated based on the achieved compression ratio, encoding/decoding times, and the entropy of the compressed data. The results are compared to a baseline compression method (e.g., zip) to assess the effectiveness of the VQ-VAE approach.

## Results

The compression algorithm was tested on a dataset of electrode recordings, and the following results were obtained:

- Average Compression Ratio: 687.95
- Average Compressed Size: 287 bytes
- Average Encoding Time: 0.0173 seconds
- Average Decoding Time: 0.0183 seconds
- Average Original Entropy: 16.83 bits/byte
- Average Decompressed Entropy: 16.63 bits/byte

These results demonstrate the effectiveness of the VQ-VAE-based compression algorithm in achieving high compression ratios while maintaining relatively low encoding and decoding times. The compressed data size is significantly reduced compared to the original recordings, enabling efficient wireless transmission.

## Usage

To use the compression algorithm, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/ShaliniAnandaPhD/brain-electrode-compression.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your electrode recording WAV files in the `data/` directory.

4. Run the compression script:
   ```
   python src/vqvae_compression.py
   ```

5. The compressed data will be saved in the `results/` directory, along with a summary of the compression results.

For more detailed information on the compression approach and experimental setup, please refer to the `docs/APPROACH.md` file.

## Contributing

Contributions to this project are welcome. If you have any ideas for improvements or would like to report a bug, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
