#%%
import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributions import Normal

from utils import default_args, init_weights, var, sample, generate_2d_sinusoidal_positions, Ted_Conv2d



class Generator(nn.Module):
    def __init__(self, args = default_args):
        super(Generator, self).__init__()
        
        self.args = args
                        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 5 * 5))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 5 * 5),
            nn.Softplus())
        
        self.a = nn.Sequential(
            
            # 5 by 5
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [1, 1, 3, 3]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [1, 1, 3, 3]),  
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 10 by 10
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [1, 3, 3, 5]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [1, 3, 3, 5]),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 20 by 20
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [3, 3, 5, 5]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [3, 3, 5, 5]),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 40 by 40
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [3, 5, 5, 7]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [3, 5, 5, 7]),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear",
                align_corners = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
            # 80 by 80
            )
                
        self.b = nn.Sequential(
            Ted_Conv2d(
                32 + self.args.pos_channels,
                [32 // 4] * 4,
                kernel_sizes = [5, 5, 7, 7]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                32, 
                3, 
                kernel_size=1, 
                padding=0),  
            nn.Tanh())
        
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, seeds = None, use_std = True):
        if(seeds == None):
            seeds = torch.stack([torch.randn(self.args.seed_size) for _ in range(self.args.batch_size)], dim = 0).to(self.args.device)
        
        mu, std = var(seeds, self.mu, self.std, self.args)
        if(use_std):
            sampled = sample(mu, std, self.args.device)
        else:
            sampled = sample(mu, 0 * std, self.args.device)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)    
            
        out = action.view(-1, 32, 5, 5)
        out = self.a(out)
        
        if(self.args.pos_channels != 0):
            positional_layers = generate_2d_sinusoidal_positions(
                batch_size = out.shape[0], 
                image_size = out.shape[2], 
                d_model = self.args.pos_channels,
                device=self.args.device)
            out = torch.cat([out, positional_layers], dim = 1)
        
        out = self.b(out)
        
        out = (out + 1) / 2
        
        crop = 8
        width, height = out.shape[-2], out.shape[-1]
        out = out[:, :, crop:width-crop, crop:height-crop]
        return out, log_prob



if(__name__ == "__main__"):
    args = default_args
    gen = Generator(args)
    print("\n\n")
    print(gen)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(gen, (args.batch_size, default_args.seed_size)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%
