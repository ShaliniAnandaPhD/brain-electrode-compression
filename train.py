import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import HybridModel
from dataset import AudioDataset
from utils import hybrid_loss

def train_hybrid_model(model, dataloader, epochs=100, lr=1e-3):
    """
    Train the HybridModel.

    Args:
        model (HybridModel): The HybridModel instance to train.
        dataloader (DataLoader): The DataLoader containing the training data.
        epochs (int): The number of training epochs (default: 100).
        lr (float): The learning rate for the optimizer (default: 1e-3).

    Returns:
        list: The training losses for each epoch.
    """
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

