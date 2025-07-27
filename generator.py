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
from utils_for_torch import init_weights, var, sample, CNN_Attention_Blend



# Let's make a generator!
class Generator(nn.Module):
    def __init__(self, args = default_args):
        super(Generator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to get the number of channels layers should have.
        example = torch.zeros(self.args.batch_size, self.args.seed_size)
        print("GEN START:", example.shape)
                     
        # From seeds to tensor for CNN.
        self.process_seeds = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 4 * 4))
        
        example = self.process_seeds(example).view(-1, 32, 4, 4)
        print("GEN process_seeds:", example.shape)
        
        # CNNs growing image size.
        self.a = nn.Sequential(
            # 4 by 4
            CNN_Attention_Blend(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow = True,
                shrink = False,
                paying_attention = False, 
                attention_kernel_size = 1,
                args = default_args),
            # 8 by 8           
            )
        
        example = self.a(example)
        print("GEN a:", example.shape)
        
        # Mean and standard deviation.
        self.mu = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow = False,
                shrink = False,
                paying_attention = False, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args))
        
        self.std = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow = False,
                shrink = False,
                paying_attention = False, 
                attention_kernel_size = 3,
                activations = False, 
                args = default_args),
            nn.Softplus())
        
        example_mu = self.mu(example)
        example_std = self.std(example)
        example = sample(example_mu, example_std)
        print("GEN a:", example.shape)
            
        # CNNs growing image. 
        self.b = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow = True,
                shrink = False,
                paying_attention = False, 
                attention_kernel_size = 3,
                args = default_args))
            # 16 by 16
            
        example = self.b(example)
        print("GEN b:", example.shape)
            
        channels_for_pos = 3
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        self.c = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 5, 
                grow = True,
                shrink = False,
                paying_attention = True, 
                attention_kernel_size = 5,
                args = default_args))
            # 32 by 32
            
        channels_for_pos = 3
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        self.d = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 7, 
                grow = True,
                shrink = False,
                paying_attention = True, 
                attention_kernel_size = 5,
                args = default_args))
            # 64 by 64     

        # CNNs growing image and finishing image. 
        channels_for_pos = 3
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        self.finish = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 7, 
                grow = False,
                shrink = False,
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
            sampled = sample(mu, std)
        else:
            sampled = sample(mu, 0 * std)
        
        # Grow.
        b = self.b(sampled)
        
        # Add position layers.
        pos_16 = self.learned_pos_16.repeat(b.shape[0], 1, 1, 1)
        pos_16 = F.interpolate(pos_16, scale_factor = 2, mode = "bilinear", align_corners = True)
        b = torch.cat([b, pos_16], dim = 1)
        
        # Grow.
        c = self.c(b)
        
        # Add position layers.
        pos_32 = self.learned_pos_32.repeat(b.shape[0], 1, 1, 1)
        pos_32 = F.interpolate(pos_32, scale_factor = 2, mode = "bilinear", align_corners = True)
        c = torch.cat([c, pos_32], dim = 1)
        
        # Grow.
        d = self.d(c)
        
        # Add position layers.
        pos_64 = self.learned_pos_64.repeat(b.shape[0], 1, 1, 1)
        pos_64 = F.interpolate(pos_64, scale_factor = 4, mode = "bilinear", align_corners = True)
        d = torch.cat([d, pos_64], dim = 1)
        
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
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%