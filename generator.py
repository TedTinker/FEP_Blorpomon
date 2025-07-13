#%%

import os 

os.chdir(r"C:\Users\Ted\Desktop\FEP_Blorpomon")

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from utils import default_args, init_weights, var, sample, My_Layer, position_layers



# Let's make a generator!
class Generator(nn.Module):
    def __init__(self, args = default_args):
        super(Generator, self).__init__()
        
        self.args = args
                     
        # From seeds to tensor for CNN.
        self.process_seeds = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 4 * 4))
        
        # CNNs growing image size.
        self.a = nn.Sequential(
            # 4 by 4
            My_Layer(
                in_channels = 34, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = False, 
                attention_kernel_size = 1,
                args = default_args),
            # 8 by 8
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            # 16 by 16
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args))
            # 32 by 32
        
        # Mean and standard deviation.
        self.mu = nn.Sequential(
            My_Layer(
                in_channels = 34, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "none", 
                paying_attention = True, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args))
        
        self.std = nn.Sequential(
            My_Layer(
                in_channels = 34, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "none", 
                paying_attention = True, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args),
            nn.Softplus())
            
        # CNNs growing image and finishing image. 
        self.b = nn.Sequential(
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            # 64 by 64                
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Finish
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size=1, 
                padding=0),  
            nn.Tanh())
        
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, seeds = None, use_std = True):
        # Start with seeds.
        if(seeds == None):
            seeds = torch.stack([torch.randn(self.args.seed_size) for _ in range(self.args.batch_size)], dim = 0).to(self.args.device)
        
        processed_seeds = self.process_seeds(seeds)
        processed_seeds = processed_seeds.view(-1, 32, 4, 4)
        
        # Add position layers.
        h_grad, v_grad = position_layers(processed_seeds)
        processed_seeds = torch.cat([processed_seeds, h_grad, v_grad], dim = 1)
        
        # Grow.
        a = self.a(processed_seeds)
        
        # Add position layers.
        h_grad, v_grad = position_layers(a)
        a = torch.cat([a, h_grad, v_grad], dim = 1)
            
        # Apply mean and standard deviation.
        mu, std = var(a, self.mu, self.std, self.args)
        if(use_std and self.args.alpha != 0):
            sampled = sample(mu, std, self.args.device)
        else:
            sampled = sample(mu, 0 * std, self.args.device)
        
        # Finish.
        out = self.b(sampled)
        out = (out + 1) / 2
        
        return out, mu, std



# Let's check it out!
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