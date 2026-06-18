from rae_faces import *
import torch
import numpy as np

latent_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RAE(latent_dim).to(device).eval()
state = torch.load("../rae/model_faces_d100.pth", map_location=device)
model.load_state_dict(state) # oder state, falls nur Gewichte

beste = [0] * latent_dim

for i in range(latent_dim):
    beste[i] = (i, 0) # (index, wichtigkeit)
    z_arr = np.array([0.5] * latent_dim, dtype=np.float32)

    z_arr[i] = 0.0 # oder 1.0, je nachdem was
    z = torch.tensor(z_arr).unsqueeze(0).to(device) # [1, latent_dim]
    output_0 = model.decoder(z).detach().squeeze().cpu().numpy().transpose(1,2,0)

    z_arr[i] = 1.0  # oder 1.0, je nachdem was
    z = torch.tensor(z_arr).unsqueeze(0).to(device)  # [1, latent_dim]
    output_1 = model.decoder(z).detach().squeeze().cpu().numpy().transpose(1, 2, 0)

    diff = np.abs(output_1 - output_0).mean()  # Durchschnittliche Änderung über alle Pixel/Kanäle
    beste[i] = (i, diff)

beste.sort(key=lambda x: x[1], reverse=True)
with open("wichtigste_latents_rae.txt", "w") as f:
    text = str(beste[0][0])
    for idx in beste[1:]:
        text += "," + str(idx[0])
    f.write(text)
print(beste)