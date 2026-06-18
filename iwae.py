import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class IWAE(nn.Module): # Umbenannt zur Klarheit
    def __init__(self, latent_dim=100):
        super(IWAE, self).__init__()
        self.latent_dim = latent_dim

        # [Encoder bleibt identisch zu deinem Code]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1),
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

        # Decoder: Latent Space -> 3x32x32 [Bleibt identisch]
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 192 * 4 * 4),
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (192, 4, 4)),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar, num_samples=10):
        # Erzeuge num_samples pro Batch-Element -> Form: (num_samples, batch_size, latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, mu.size(0), self.latent_dim, device=mu.device)
        return mu + eps * std  # Nutzen von Broadcasting

    def decode(self, z):
        # z hat hier die Form (num_samples * batch_size, latent_dim) beim Training
        out = self.decoder_input(z)
        out = self.decoder(out)
        return out

    def forward(self, x, num_samples=10):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # z Form: (num_samples, batch_size, latent_dim)
        z = self.reparameterize(mu, logvar, num_samples)
        
        # Flachdrücken für den Decoder: (num_samples * batch_size, latent_dim)
        z_flat = z.view(num_samples * x.size(0), self.latent_dim)
        recon_flat = self.decode(z_flat)
        
        # Zurückformen in (num_samples, batch_size, C, H, W)
        recon_x = recon_flat.view(num_samples, x.size(0), x.size(1), x.size(2), x.size(3))
        
        return recon_x, mu, logvar, z


def iwae_loss_function(recon_x, x, mu, logvar, z, num_samples=10):
    """
    Berechnet den IWAE Loss mittels des Log-Sum-Exp Tricks für numerische Stabilität.
    """
    
    # 1. Rekonstruktions-Loss für alle Samples parallel berechnen
    # Form von x erweitern auf (num_samples, batch_size, C, H, W)
    x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    
    # Da du im Original MSE mit reduction='sum' nutzt, berechnen wir hier den MSE pro Sample & Batch-Element
    # Form danach: (num_samples, batch_size)
    log_p_x_given_z = -F.mse_loss(recon_x, x_expanded, reduction='none').sum(dim=[2, 3, 4])

    # 2. Log-Wahrscheinlichkeiten für Prior p(z) und vorschlagende Verteilung q(z|x)
    # p(z) ist Standardnormalverteilung N(0, I)
    log_p_z = -0.5 * torch.sum(z ** 2 + torch.log(torch.tensor(2 * torch.pi, device=z.device)), dim=-1)
    
    # q(z|x) ist N(mu, logvar)
    mu_expanded = mu.unsqueeze(0)
    logvar_expanded = logvar.unsqueeze(0)
    log_q_z_given_x = -0.5 * torch.sum(
        logvar_expanded + torch.log(torch.tensor(2 * torch.pi, device=z.device)) + ((z - mu_expanded) ** 2 / torch.exp(logvar_expanded)), 
        dim=-1
    )

    # 3. Importance Weights im Log-Space: log(w) = log p(x,z) - log q(z|x)
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

    # 4. Log-Sum-Exp Trick zur Berechnung von: log(1/k * sum(exp(log_w)))
    # Wir nehmen den Mittelwert über die Samples (dim=0) im Log-Space
    loss = -torch.sum(torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float, device=z.device)))

    return loss

from vae_gemini_deeper import *

if __name__ == "__main__":

    num_samples = 10
    model = IWAE(latent_dim=latent_dim).to(device)
    model_path = "iwae.pth"
    #if os.path.exists(model_path):
        #model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    l = len(dataloader)

    vae_losses = []
    vae_grad_norms = []
    vae_cos_sims = []
    previous_epoch_grad_sum_flat = None

    for ep in range(100):

        epoch_loss_sum = 0.0
        current_epoch_grad_sum_flat = None

        for i, batch in enumerate(dataloader):
            print(i / l)

            model.train()
            data = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, z = model(data)
            loss = iwae_loss_function(recon_batch, data, mu, logvar, z, num_samples=num_samples)

            loss.backward()

            epoch_loss_sum += loss.item()

            # Collect gradients for this batch before optimizer step
            current_grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            if current_grads:
                batch_grad_flat = torch.cat(current_grads)
                if current_epoch_grad_sum_flat is None:
                    current_epoch_grad_sum_flat = batch_grad_flat.detach().clone()
                else:
                    current_epoch_grad_sum_flat += batch_grad_flat.detach().clone()

            optimizer.step()

        torch.save(model.state_dict(), model_path)
        avg_epoch_loss = epoch_loss_sum / len(dataloader)
        vae_losses.append(avg_epoch_loss)

        print(f"Epoch [{ep + 1}/1000], Avg Loss: {avg_epoch_loss:.4f}")

        if current_epoch_grad_sum_flat is not None:
            epoch_total_norm = current_epoch_grad_sum_flat.norm(2)
            vae_grad_norms.append(epoch_total_norm.item())

            if previous_epoch_grad_sum_flat is not None:
                cosine_similarity = F.cosine_similarity(current_epoch_grad_sum_flat, previous_epoch_grad_sum_flat, dim=0)
                vae_cos_sims.append(cosine_similarity.item())

            previous_epoch_grad_sum_flat = current_epoch_grad_sum_flat.detach().clone()

        number_tests = 2
        z = torch.randn((number_tests, latent_dim)).to(device)
        imgs = model.decode(z).detach().cpu().numpy().transpose(0, 2, 3, 1)
        import matplotlib.pyplot as plt

        for i in range(number_tests):
            plt.title(f"Epoch {ep}")
            plt.imshow(imgs[i])
            plt.show()

    plt.plot(range(len(vae_losses)), vae_losses)
    plt.title("VAE Losses")
    plt.savefig("iwae_losses.png")
    plt.show()

    plt.plot(range(len(vae_grad_norms)), vae_grad_norms)
    plt.title("VAE Grad Norms")
    plt.savefig("iwae_grad_norms.png")
    plt.show()

    plt.plot(range(len(vae_cos_sims)), vae_cos_sims)
    plt.title("VAE Cos Sims")
    plt.savefig("iwae_cos_sims.png")
    plt.show()