import os 
from utils import file_location

os.chdir(file_location)

import torch.nn.functional as F


import torch 
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from utils import get_random_batch, default_args
from utils_for_torch import init_weights, var, sample, My_Layer, get_stats, add_position_layers



# Let's make a discriminator!
class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to get the number of channels layers should have.
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        
        channels_for_pos = 3
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        pos_64 = self.learned_pos_64.repeat(example.shape[0], 1, 1, 1)
        pos_64 = F.interpolate(pos_64, scale_factor = 4, mode = "bilinear", align_corners = True)
        example = torch.cat([example, pos_64], dim = 1)
        
        stats = get_stats(example, self.args).cpu()
        stat_channels = stats.shape[1]
                        
        # Process statistics.
        self.stats = nn.Sequential(
            nn.Conv2d(
                in_channels = stat_channels, 
                out_channels = 32,
                kernel_size = 7,
                padding = 3,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        stats = self.stats(stats)
        
        # Process images.
        self.images = nn.Sequential(
            nn.Conv2d(
                in_channels = 3 + channels_for_pos, 
                out_channels = 32,
                kernel_size = 7,
                padding = 3,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        example = self.images(example)
        example = torch.cat([example, stats], dim = 1)
        
        # CNNs shrinking image size.
        self.a = nn.Sequential(
            # 64 by 64
            nn.Dropout2d(p=self.args.dropout),
            My_Layer(
                in_channels = 64, 
                channels = 32, 
                kernel_size = 7, 
                grow_or_shrink = "shrink", 
                paying_attention = False, 
                attention_kernel_size = 5,
                args = default_args),
            nn.Dropout2d(p=self.args.dropout))
        
        example = self.a(example)
        channels_for_pos = 3
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        pos_32 = self.learned_pos_32.repeat(example.shape[0], 1, 1, 1)
        pos_32 = F.interpolate(pos_32, scale_factor = 2, mode = "bilinear", align_corners = True)
        example = torch.cat([example, pos_32], dim = 1)
        
        self.b = nn.Sequential(
            # 32 by 32
            nn.Dropout2d(p=self.args.dropout),
            My_Layer(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 7, 
                grow_or_shrink = "shrink", 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            nn.Dropout2d(p=self.args.dropout),)
            # 16 by 16
            
        example = self.b(example)
        channels_for_pos = 3
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        pos_16 = self.learned_pos_16.repeat(example.shape[0], 1, 1, 1)
        pos_16 = F.interpolate(pos_16, scale_factor = 2, mode = "bilinear", align_corners = True)
        example = torch.cat([example, pos_16], dim = 1)
            
        self.c = nn.Sequential(
            My_Layer(
                in_channels = 32 + channels_for_pos, 
                channels = 32, 
                kernel_size = 5, 
                grow_or_shrink = "shrink", 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            # 8 by 8
            nn.Dropout2d(p=self.args.dropout),
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "shrink", 
                paying_attention = False, 
                attention_kernel_size = 1,
                args = default_args))
            # 4 by 4
                
        example = self.c(example).view(self.args.batch_size, -1)
        
        # Flatten.
        self.d = nn.Sequential(
            nn.Linear(example.shape[-1], self.args.inner_state_size))
        
        # Mean and standard deviation.
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.inner_state_size, 
                out_features = 1))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.inner_state_size, 
                out_features = 1),
            nn.Softplus())
        
        self.apply(init_weights)
        self.to(self.args.device)
        
        

    def forward(self, images):
        batch_size, num_channels, height, width = images.size()
        images = (images * 2) - 1
        
        # Add position layers.
        pos_64 = self.learned_pos_64.repeat(images.shape[0], 1, 1, 1)
        pos_64 = F.interpolate(pos_64, scale_factor = 4, mode = "bilinear", align_corners = True)
        images = torch.cat([images, pos_64], dim = 1)
                        
        # Process statistics and images.
        stats = get_stats(images, self.args)
        stats = self.stats(stats)
        images = self.images(images)
        images = torch.cat([images, stats], dim = 1)
    
        # Shrinking and flattening.
        a = self.a(images)
        
        # Add position layers.
        pos_32 = self.learned_pos_32.repeat(a.shape[0], 1, 1, 1)
        pos_32 = F.interpolate(pos_32, scale_factor = 2, mode = "bilinear", align_corners = True)
        a = torch.cat([a, pos_32], dim = 1)
        b = self.b(a)
        
        # Add position layers.
        pos_16 = self.learned_pos_16.repeat(a.shape[0], 1, 1, 1)
        pos_16 = F.interpolate(pos_16, scale_factor = 2, mode = "bilinear", align_corners = True)
        b = torch.cat([b, pos_16], dim = 1)
        c = self.c(b)
        
        # Flatten.
        d = self.d(c.view(batch_size, -1))
        
        # Apply mean and standard deviation.
        mu, std = var(d, self.mu, self.std, self.args)
        sampled = sample(mu, std, self.args.device)
        sampled = torch.tanh(sampled)
        
        # Finish.
        out = (sampled + 1) / 2
        return out, mu, std



# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    dis = Discriminator(args)
    print("\n\n")
    print(dis)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dis, (args.batch_size, 3, args.image_size, args.image_size)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
    

    