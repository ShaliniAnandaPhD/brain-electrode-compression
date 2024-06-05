# model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define the VQ-VAE architecture
class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )
        self.commitment_cost = commitment_cost

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def quantize(self, encoding_indices):
        return self.codebook(encoding_indices)

    def forward(self, x):
        z = self.encode(x)
        encoding_indices = torch.argmin(torch.sum((z.unsqueeze(1) - self.codebook.weight) ** 2, dim=2), dim=1)
        z_q = self.quantize(encoding_indices)
        commitment_loss = self.commitment_cost * torch.mean((z_q.detach() - z) ** 2)
        z_q = z + (z_q - z).detach()
        x_recon = self.decode(z_q)
        return x_recon, z, encoding_indices, commitment_loss

# Define the Convolutional Autoencoder (CAE) architecture
class CAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Define the HybridModel architecture
class HybridModel(nn.Module):
    def __init__(self, input_dim, latent_dim_vqvae, latent_dim_cae, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(HybridModel, self).__init__()
        self.vqvae = VQVAE(input_dim, latent_dim_vqvae, num_embeddings, embedding_dim, commitment_cost)
        self.cae = CAE(latent_dim_vqvae, latent_dim_cae)

    def forward(self, x):
        x_recon_vqvae, z_vqvae, encoding_indices, commitment_loss = self.vqvae(x)
        z_cae = self.cae.encoder(x_recon_vqvae.unsqueeze(1))  # Ensure correct shape
        x_recon_cae = self.cae.decoder(z_cae)
        return x_recon_cae.squeeze(1), x_recon_vqvae, z_vqvae, encoding_indices, commitment_loss

# Define the hybrid loss function
def hybrid_loss(model, x, tilde_x, hat_x, z_e, encoding_indices, commitment_loss, alpha=1.0):
    recon_loss_vqvae = nn.functional.mse_loss(hat_x, x)
    recon_loss_cae = nn.functional.mse_loss(tilde_x, x)
    z_q = model.vqvae.quantize(encoding_indices)
    vq_loss = torch.mean((z_e.detach() - z_q.float()) ** 2)
    commitment_loss = torch.mean(commitment_loss)
    total_loss = recon_loss_cae + alpha * (recon_loss_vqvae + vq_loss + commitment_loss)
    return total_loss

# Training loop
def train_hybrid_model(model, dataloader, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.float()
            optimizer.zero_grad()
            x_recon_cae, x_recon_vqvae, z_vqvae, encoding_indices, commitment_loss = model(data)
            loss = hybrid_loss(model, data, x_recon_cae, x_recon_vqvae, z_vqvae, encoding_indices, commitment_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses
