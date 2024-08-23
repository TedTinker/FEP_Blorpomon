#%%
import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributions import Normal

from utils import default_args, init_weights, var, sample, generate_2d_sinusoidal_positions



class Generator(nn.Module):
    def __init__(self, args = default_args):
        super(Generator, self).__init__()
        
        self.args = args
                        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 4 * 4))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 4 * 4),
            nn.Softplus())
        
        self.a = nn.Sequential(
            nn.Conv2d(
                32 + self.args.pos_channels, 
                32, 
                kernel_size=3, 
                padding=1),  
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                32, 
                32, 
                kernel_size=5, 
                padding=2), 
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                32, 
                32, 
                kernel_size=5, 
                padding=2), 
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  
            nn.Tanh())
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, z = None, use_std = True):
        if(z == None):
            z = torch.stack([torch.randn(self.args.seed_size) for _ in range(self.args.batch_size)], dim = 0).to(self.args.device)
        
        mu, std = var(z, self.mu, self.std, self.args)
        if(use_std):
            sampled = sample(mu, std, self.args.device)
        else:
            sampled = sample(mu, 0 * std, self.args.device)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)    
            
        out = action.view(-1, 32, 4, 4)
        
        positional_layers = generate_2d_sinusoidal_positions(
            batch_size = out.shape[0], 
            image_size = out.shape[2], 
            d_model = self.args.pos_channels,
            device=self.args.device)
        out = torch.cat([out, positional_layers], dim = 1)
        
        
        out = self.a(out)
        out = (out + 1) / 2
        return out, log_prob



if(__name__ == "__main__"):
    gen = Generator()
    print("\n\n")
    print(gen)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(gen, (16, default_args.seed_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%
