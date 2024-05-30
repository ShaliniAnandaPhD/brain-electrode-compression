import os
import numpy as np
import torch
import torch.optim as optim
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from model import VQVAE, vqvae_loss

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Segment the audio file into smaller segments
def segment_audio(file_path, segment_size):
    audio_data, samplerate = sf.read(file_path)
    segments = [audio_data[i:i + segment_size] for i in range(0, len(audio_data), segment_size)]
    for i in range(len(segments)):
        if len(segments[i]) < segment_size:
            segments[i] = np.pad(segments[i], (0, segment_size - len(segments[i])), 'constant')
    return segments, samplerate

# Dataset class for loading audio segments
class AudioDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]

# Train the VQ-VAE model
def train_vqvae(model, dataloader, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0
        for data in dataloader:
            data = data.float().view(data.size(0), -1)  # Flatten the input data
            optimizer.zero_grad()
            recon_batch, z, z_q = model(data)
            loss = vqvae_loss(recon_batch, data, z, z_q)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(dataloader.dataset)}')

# Example usage
data_dir = 'data'
segment_size = 16000
latent_dim = 2  # Further reduced latent dimension for less redundancy
num_embeddings = 128  # Reduced number of embeddings for more efficient quantization
embedding_dim = latent_dim

# Get the .wav files in the directory
wav_files = [file for file in os.listdir(data_dir) if file.endswith('.wav')]

# Train the VQ-VAE model on the .wav files
segments_list = []
samplerate_list = []
for file in wav_files:
    file_path = os.path.join(data_dir, file)
    segments, samplerate = segment_audio(file_path, segment_size)
    segments_list.extend(segments)
    samplerate_list.append(samplerate)

audio_dataset = AudioDataset(torch.tensor(segments_list, dtype=torch.float32))
dataloader = DataLoader(audio_dataset, batch_size=64, shuffle=True)

model = VQVAE(segment_size, latent_dim, num_embeddings, embedding_dim)
train_vqvae(model, dataloader, epochs=100)
torch.save(model.state_dict(), 'vqvae_audio_compression.pth')

