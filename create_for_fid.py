from iwae import *
import torch
import numpy as np
import os
from PIL import Image

latent_dim = 100
model = VAE(latent_dim).to(device)
model.load_state_dict(torch.load("beta-vae_1.2828282828282829.pth"))

number_tests = 5000
z = torch.randn((number_tests, latent_dim)).to(device)
imgs = model.decode(z).detach().cpu().numpy().transpose(0, 2, 3, 1)

output_dir = "generated/beta-vae"

for i in range(number_tests):
    img_array = (imgs[i] * 255).astype(np.uint8)

    # Speichere als PNG
    img = Image.fromarray(img_array, mode='RGB')
    img.save(os.path.join(output_dir, f"generated_{i:05d}.png"))

    if (i + 1) % 500 == 0:
        print(f"Gespeichert: {i + 1}/{number_tests}")

print(f"✓ Alle {number_tests} Bilder in '{output_dir}' gespeichert")