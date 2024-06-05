# model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class VQVAE(nn.Module):
    def __init__(self, segment_size, latent_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(segment_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, segment_size)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        z_q, _ = self.vector_quantize(z)
        x_recon = self.decode(z_q)
        return x_recon, z, z_q
    
    def vector_quantize(self, z):
        distances = (z.unsqueeze(1) - self.codebook.weight).pow(2).sum(dim=-1)
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices)
        return z_q, indices

class AudioDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]

def vqvae_loss(x_recon, x, z, z_q, commitment_cost=0.25):
    recon_loss = nn.MSELoss()(x_recon, x)
    vq_loss = nn.MSELoss()(z_q, z.detach())
    commitment_loss = commitment_cost * nn.MSELoss()(z, z_q.detach())
    return recon_loss + vq_loss + commitment_loss

def train_vqvae(model, dataloader, epochs, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, x in enumerate(dataloader):
            x = x.float()
            optimizer.zero_grad()
            x_recon, z, z_q = model(x)
            loss = vqvae_loss(x_recon, x, z, z_q)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (i + 1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def compress_audio(model, segments):
    compressed_audio = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment_tensor = torch.tensor(segment).float().unsqueeze(0)
            z = model.encode(segment_tensor)
            _, indices = model.vector_quantize(z)
            compressed_audio.append(indices.squeeze().numpy())
    compressed_audio = np.array(compressed_audio)
    return compressed_audio

def decompress_audio(model, compressed_audio, segment_size):
    decompressed_audio = []
    model.eval()
    with torch.no_grad():
        for indices in compressed_audio:
            indices_tensor = torch.tensor(indices).unsqueeze(0)
            z_q = model.codebook(indices_tensor)
            segment_recon = model.decode(z_q)
            decompressed_audio.append(segment_recon.squeeze().numpy())
    decompressed_audio = np.concatenate(decompressed_audio)
    return decompressed_audio[:segment_size * len(compressed_audio)]

def entropy_encode(indices):
    indices_bytes = indices.tobytes()
    compressed_indices = bz2.compress(indices_bytes)
    return compressed_indices

def entropy_decode(encoded_indices):
    decompressed_indices = bz2.decompress(encoded_indices)
    indices = np.frombuffer(decompressed_indices, dtype=np.int64)
    return indices
