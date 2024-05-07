import torch
import torch.nn as nn

global_dim = 512


# Define a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x  # Store the input
        x = self.linear1(x)
        x = nn.ReLU(True)(x)
        x = self.linear2(x)
        x += residual  # Add the input back to the output
        x = nn.ReLU(True)(x)
        return x


class EncoderBrain(nn.Module):
    def __init__(self, flags):
        super(EncoderBrain, self).__init__()
        self.flags = flags
        self.hidden_dim = global_dim

        # Initial Linear Layer
        self.linear_in = nn.Linear(flags.m1_dim, self.hidden_dim)

        # Residual Blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.hidden_dim) for _ in range(flags.num_hidden_layers - 1)]
        )

        # Output Layers
        self.hidden_mu = nn.Linear(
            in_features=self.hidden_dim, out_features=flags.class_dim, bias=True
        )
        self.hidden_logvar = nn.Linear(
            in_features=self.hidden_dim, out_features=flags.class_dim, bias=True
        )

    def forward(self, x):
        h = self.linear_in(x)
        h = nn.ReLU(True)(h)  # Apply ReLU after the first linear layer

        # Pass through residual blocks
        for block in self.residual_blocks:
            h = block(h)

        h = h.view(h.size(0), -1)  # Flatten for output layers
        latent_space_mu = self.hidden_mu(h)
        latent_space_logvar = self.hidden_logvar(h)
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)
        return None, None, latent_space_mu, latent_space_logvar


class DecoderBrain(nn.Module):
    def __init__(self, flags):
        super(DecoderBrain, self).__init__()
        self.flags = flags
        self.hidden_dim = global_dim
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
        self.fc3 = nn.Linear(self.hidden_dim, flags.m1_dim)
        self.relu = nn.ReLU()

    def forward(self, style_latent_space, class_latent_space):
        z = class_latent_space
        x_hat = self.dec(z)
        x_hat = self.fc3(x_hat)
        return x_hat, torch.tensor(0.75).to(z.device)
