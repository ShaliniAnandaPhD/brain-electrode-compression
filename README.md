Brain Electrode Compression

This repository contains a compression algorithm for brain electrode recordings, specifically designed for the N1 implant. The algorithm aims to achieve high compression ratios while maintaining low latency and power consumption, enabling efficient wireless transmission of electrode data.

Overview

The N1 implant, located in the motor cortex of a non-human primate, generates approximately 200Mbps of electrode data (1024 electrodes @ 20kHz, 10b resolution) during video game play. However, the implant can only transmit around 1Mbps wirelessly. To overcome this limitation, a compression algorithm is necessary to reduce the data size by more than 200 times while operating in real-time (< 1ms) and consuming low power (< 10mW, including radio).

This repository explores a novel approach to compressing brain electrode recordings using a Vector Quantized Variational Autoencoder (VQ-VAE) architecture.

Requirements

- Python 3.x
- NumPy
- SoundFile
- PyTorch
- SciPy

Usage

1. Clone the repository:
   ```
   git clone https://github.com/ShaliniAnandaPhD/brain-electrode-compression.git
   cd brain-electrode-compression
   ```

2. Install the required dependencies and train the VQ-VAE model:
   ```
   ./build.sh
   ```

3. Encode audio files:
   ```
   ./encode.py path/to/input/audio.wav path/to/output/encoded.bin
   ```

4. Decode audio files:
   ```
   ./decode.py path/to/input/encoded.bin path/to/output/decoded.wav
   ```

Scripts

- encode.py: Script for encoding audio files using the trained VQ-VAE model.
- decode.py: Script for decoding compressed audio files using the trained VQ-VAE model.
- train.py: Script for training the VQ-VAE model on the audio dataset.
- model.py: Contains the implementation of the VQ-VAE model and related utility functions.
- build.sh: Shell script for setting up the environment, installing dependencies, and training the VQ-VAE model.

Evaluation

To verify compression is lossless and measure compression ratio:
```
./eval.sh
```

License

This project is licensed under the MIT License.

