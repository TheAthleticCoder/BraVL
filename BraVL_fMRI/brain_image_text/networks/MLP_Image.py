import torch
import torch.nn as nn

mlp_image_dim = 2048


class EncoderImage(nn.Module):
    def __init__(self, flags):
        super(EncoderImage, self).__init__()
        self.flags = flags
        self.hidden_dim = mlp_image_dim
        self.embed_dim = flags.class_dim  # Embedding dimension for attention

        # Initial MLP layers (similar to before)
        modules = []
        modules.append(
            nn.Sequential(nn.Linear(flags.m2_dim, self.hidden_dim), nn.ReLU(True))
        )
        modules.extend(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True)
                )
                for _ in range(flags.num_hidden_layers - 1)
            ]
        )
        self.enc = nn.Sequential(*modules)

        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, batch_first=True
        )

        # Output Layers (similar to before)
        self.hidden_mu = nn.Linear(
            in_features=self.hidden_dim, out_features=flags.class_dim, bias=True
        )
        self.hidden_logvar = nn.Linear(
            in_features=self.hidden_dim, out_features=flags.class_dim, bias=True
        )

    def forward(self, x):
        h = self.enc(x)  # Pass through initial MLP layers
        h = h.view(h.size(0), -1, self.embed_dim)  # Reshape for attention

        # Apply self-attention
        attn_output, attn_weights = self.self_attn(h, h, h)

        # Combine attention output with original features (choose one solution)
        # Solution (a): Average across sequence length
        attn_output = torch.mean(attn_output, dim=1, keepdim=True)
        h = torch.cat((h, attn_output), dim=-1)

        # Solution (b): Use a linear layer to reduce dimensionality
        # h = torch.cat((h, attn_output), dim=-1)
        # h = nn.Linear(2 * self.embed_dim, self.embed_dim)(h)

        h = h.view(h.size(0), -1)  # Flatten for output layers
        latent_space_mu = self.hidden_mu(h)
        latent_space_logvar = self.hidden_logvar(h)
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)
        return None, None, latent_space_mu, latent_space_logvar


class DecoderImage(nn.Module):
    def __init__(self, flags):
        super(DecoderImage, self).__init__()
        self.flags = flags
        self.hidden_dim = mlp_image_dim
        modules = []

        modules.append(
            nn.Sequential(nn.Linear(flags.class_dim, self.hidden_dim), nn.ReLU(True))
        )

        modules.extend(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True)
                )
                for _ in range(flags.num_hidden_layers - 1)
            ]
        )
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, flags.m2_dim, bias=True)

    def forward(self, style_latent_space, class_latent_space):
        z = class_latent_space
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)
        return x_hat, torch.tensor(0.75).to(z.device)
