# import ML libraries
import torch
import torch.nn as nn
import torchvision

# import general libraries
import os
import matplotlib.pyplot as plt
import numpy

# Get the directory of the current module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# reparameterization trick
def reparametrize(mu: torch.Tensor, log_var: torch.Tensor):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

# variational decoder
class Decoder(nn.Module):

    # constructor
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # forward pass
    def forward(self, x: torch.Tensor):
        x = self.relu(self.fc(x))
        x = self.sigmoid(self.fc_out(x))
        return x

class Fashion(nn.Module):

    # constructor
    def __init__(self):
        super(Fashion, self).__init__()

        # load decoder
        self.decoder = Decoder(latent_dim=20, hidden_dim=400, output_dim=28*28)
        self.decoder.load_state_dict(
            torch.load(
                os.path.join(CURRENT_DIR, 'weights/vae_decoder.pt'), 
                map_location=torch.device('cpu'),
                weights_only = True
                ), 
                strict=False)

        # load mean values for mu and log_var
        self.mu_mean = torch.load(
            os.path.join(CURRENT_DIR, 'weights/mu_mean.pt'), 
            map_location=torch.device('cpu'),
            weights_only = True
            )
        self.log_var_mean = torch.load(
            os.path.join(CURRENT_DIR, 'weights/log_var_mean.pt'), 
            map_location=torch.device('cpu'),
            weights_only = True
            )
    
    # forward pass
    def forward(self, idx: int, visualize: bool = True):

        # values associated with index
        mu = self.mu_mean[idx]
        log_var = self.log_var_mean[idx]

        # generate
        z = reparametrize(mu, log_var)
        with torch.no_grad():
            res = self.decoder(z.view(1, -1)).view(1, 28, 28)

        if visualize:
            grid_img = torchvision.utils.make_grid(res)
            plt.imshow(grid_img.permute(1, 2, 0).numpy())
            plt.savefig(os.path.join(CURRENT_DIR, "results/res.jpg"))
            plt.show()
        
        # prepare result
        out = res.numpy()

        return out
