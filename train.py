import os
import torch
from torch.utils.data import DataLoader
from model import VQVAE, AudioDataset, train_vqvae

def main():
    data_dir = 'data'
    segment_size = 16000
    latent_dim = 2
    num_embeddings = 128
    embedding_dim = latent_dim

    # Get the list of WAV files
    wav_files = [file for file in os.listdir(data_dir) if file.endswith('.wav')]

    # Create the dataset and data loader
    segments_list = []
    for file in wav_files:
        file_path = os.path.join(data_dir, file)
        segments, _ = segment_audio(file_path, segment_size)
        segments_list.extend(segments)

    audio_dataset = AudioDataset(torch.tensor(segments_list, dtype=torch.float32))
    dataloader = DataLoader(audio_dataset, batch_size=64, shuffle=True)

    # Create the VQ-VAE model
    model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)

    # Train the model
    train_vqvae(model, dataloader, epochs=100)

    # Save the trained model
    torch.save(model.state_dict(), 'vqvae_audio_compression.pth')

if __name__ == "__main__":
    main()
