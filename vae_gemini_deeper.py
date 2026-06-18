import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 3x32x32 -> Latent Space
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),  # 128
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),  # 16
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Flatten()
        )

        # Latente Parameter
        self.fc_mu = nn.Sequential(
            nn.Linear(192 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(192 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Decoder: Latent Space -> 3x32x32
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 192 * 4 * 4),
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (192, 4, 4)),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),  # 16
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1),  # 128
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),  # 256
            nn.Sigmoid() # Pixelwerte zwischen 0 und 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        z = self.encoder(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        return mu, logvar

    def decode(self, z):
      out = self.decoder_input(z)
      out = self.decoder(out)
      return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # MSE eignet sich gut für RGB-Rekonstruktionen
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL-Divergenz Formel für Normalverteilungen
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld_loss, recon_loss, kld_loss

import os
from PIL import Image
from torch.utils.data import Dataset

class NoClassImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        # List all files in the root_dir and filter for common image extensions
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                self.image_files.append(os.path.join(root_dir, filename))
        if not self.image_files:
            raise RuntimeError(f"Found 0 image files in {root_dir}. Please ensure the directory contains images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') # Ensure 3 channels

        if self.transform:
            image = self.transform(image)
        return image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations for the images
hin_transforms = transforms.Compose([
    transforms.ToTensor() # Convert PIL Image to PyTorch Tensor
])

# Load the dataset from a folder (assuming images are in a folder named 'data' in the content directory)
# You might need to upload your image folder to /content/data or specify the correct path
dataset = NoClassImageDataset("C:\\Datasets\\faces\\resized", hin_transforms)

# Create a DataLoader
batch_size = 64 # You can adjust this batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100

if __name__ == "__main__":

    model = VAE(latent_dim=latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Gesamt-Parameter: {total_params:,}")
    model_path = "vae_gemini_bigfaces_deeper.pth"
    if os.path.exists(model_path):
      model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    l = len(dataloader)

    for ep in range(1000):
        for i, batch in enumerate(dataloader):

            print(i/l)

            model.train()
            data = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kld_loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_path)
        print(f"Epoch [{ep+1}/100], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}")

        number_tests = 2
        z = torch.randn((number_tests, latent_dim)).to(device)
        imgs = model.decode(z).detach().cpu().numpy().transpose(0, 2, 3, 1)
        import matplotlib.pyplot as plt

        for i in range(number_tests):
            plt.title(ep)
            plt.imshow(imgs[i])
            plt.show()