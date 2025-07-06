
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.latent_dim
        self.input_dim = args.input_dim
        self.num_classes = args.num_classes
        self.trained = False

        self.fc1 = nn.Linear(self.input_dim + self.num_classes, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim + self.num_classes, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x, y):
        inputs = torch.cat([x, F.one_hot(y, self.num_classes).float()], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, F.one_hot(y, self.num_classes).float()], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def loss_function(self, x, y):
        #recon_x, mu, logvar = self.forward(x, y)
        recon_logits, mu, logvar = self.forward(x, y)
        recon_x = torch.sigmoid(recon_logits)    # <-- constrain to [0,1]
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def sample(self, batch_size):
        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim).to(device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        gen_x = self.decode(z, y)
        return gen_x, y
