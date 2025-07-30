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
from utils_for_torch import init_weights, var, sample, Multi_Kernel_CAB, add_position_layers



# Let's make a generator!
class Generator(nn.Module):
    def __init__(self, args = default_args):
        super(Generator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, self.args.seed_size)
        print("\nStart of Generator:", example.shape)
                     
        # From seeds to tensor for CNN.
        self.process_seeds = nn.Sequential(
            nn.Linear(
                in_features = self.args.seed_size, 
                out_features =  32 * 4 * 4))
        
        example = self.process_seeds(example).view(-1, 32, 4, 4)
        print("Gen processed seeds:", example.shape)
        
        # CNNs growing image size.
        self.a = nn.Sequential(
            # 4 by 4
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [3], 
                grow = True,
                shrink = False,
                paying_attention = False, 
                args = default_args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
            # 8 by 8           
            )
        
        example = self.a(example)
        print("Gen a:", example.shape)
        
        # Mean and standard deviation.
        self.mu = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [3], 
                grow = False,
                shrink = False,
                paying_attention = False, 
                args = default_args))
        
        self.std = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [3], 
                grow = False,
                shrink = False,
                paying_attention = False, 
                args = default_args),
            nn.Softplus())
        
        example_mu, example_std = var(example, self.mu, self.std, self.args)
        example = sample(example_mu, example_std)
        print("Gen mu and std:", example.shape)
            
        # CNNs growing image. 
        self.b = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [3], 
                grow = True,
                shrink = False,
                paying_attention = False, 
                args = default_args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
            # 16 by 16
            
        example = self.b(example)
        channels_for_pos = 3
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        example = add_position_layers(example, self.learned_pos_16, scale = 2)
        print("Gen b:", example.shape)
        
        self.c = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [5], 
                grow = True,
                shrink = False,
                paying_attention = True, 
                attention_kernel_sizes = [5],
                args = default_args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
            # 32 by 32
            
        example = self.c(example)
        channels_for_pos = 3
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        example = add_position_layers(example, self.learned_pos_32, scale = 2)
        print("Gen c:", example.shape)
        
        self.d = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [7], 
                grow = True,
                shrink = False, 
                paying_attention = True, 
                attention_kernel_sizes = [7],
                args = default_args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
            # 64 by 64     

        example = self.d(example)
        channels_for_pos = 3
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        example = add_position_layers(example, self.learned_pos_64, scale = 4)
        print("Gen d:", example.shape)
        
        # CNNs growing image and finishing image.         
        self.finish = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [32], 
                kernel_sizes = [7], 
                grow = False,
                shrink = False,
                paying_attention = False, 
                args = default_args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # Finish
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 3,
                kernel_size = 1,
                padding = 0,
                padding_mode = "reflect"),
            nn.Tanh())
        
        example = self.finish(example)
        print("Finished Generator:", example.shape, "\n")
        
        
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
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%