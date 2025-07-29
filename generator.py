#%%

import os 
from utils import file_location

os.chdir(file_location)

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from utils import default_args
from utils_for_torch import init_weights, var, sample, My_Layer, add_position_layers



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
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = False, 
                attention_kernel_size = 1,
                args = default_args),
            # 8 by 8           
            )
        
        # Mean and standard deviation.
        self.mu = nn.Sequential(
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "none", 
                paying_attention = False, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args))
        
        self.std = nn.Sequential(
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "none", 
                paying_attention = False, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args),
            nn.Softplus())
            
        # CNNs growing image. 
        self.b = nn.Sequential(
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "grow", 
                paying_attention = False, 
                attention_kernel_size = 3,
                args = default_args))
            # 16 by 16
            
        channels_for_pos = 3
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        self.c = nn.Sequential(
            My_Layer(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 5, 
                grow_or_shrink = "grow", 
                paying_attention = True, 
                attention_kernel_size = 5,
                args = default_args))
            # 32 by 32
            
        channels_for_pos = 3
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        self.d = nn.Sequential(
            My_Layer(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 7, 
                grow_or_shrink = "grow", 
                paying_attention = True, 
                attention_kernel_size = 5,
                args = default_args))
            # 64 by 64     

        # CNNs growing image and finishing image. 
        channels_for_pos = 3
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        self.finish = nn.Sequential(
            My_Layer(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 7, 
                grow_or_shrink = "none", 
                paying_attention = False, 
                attention_kernel_size = 5,
                args = default_args),
            # Finish
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 3,
                kernel_size = 1,
                padding = 0,
                padding_mode = "reflect"),
            nn.Tanh())
        
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, seeds = None, use_std = True):
        # Start with seeds.
        if(seeds == None):
            seeds = torch.randn(self.args.batch_size, self.args.seed_size, device=self.args.device)
        
        processed_seeds = self.process_seeds(seeds)
        processed_seeds = processed_seeds.view(-1, 32, 4, 4)
                
        # Grow.
        a = self.a(processed_seeds)
                    
        # Apply mean and standard deviation.
        mu, std = var(a, self.mu, self.std, self.args)
        if(use_std): 
            sampled = sample(mu, std, self.args.device)
        else:
            sampled = sample(mu, 0 * std, self.args.device)
        
        # Grow.
        b = self.b(sampled)
        b = add_position_layers(b, self.learned_pos_16, scale = 2)
        
        # Grow.
        c = self.c(b)
        c = add_position_layers(c, self.learned_pos_32, scale = 2)
        
        # Grow.
        d = self.d(c)
        d = add_position_layers(d, self.learned_pos_64, scale = 4)
        
        # Finish.
        out = self.finish(d)
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