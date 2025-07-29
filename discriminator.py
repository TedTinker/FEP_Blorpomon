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
from utils_for_torch import init_weights, var, sample, CNN_Attention_Blend, get_stats, add_position_layers



# Let's make a discriminator!
class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        print("\nStart of Discriminator:", example.shape)
        
        channels_for_pos = 3
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        example = add_position_layers(example, self.learned_pos_64, scale = 4)
        
        # Process images.
        self.images = nn.Sequential(
            nn.Conv2d(
                in_channels = example.shape[1], 
                out_channels = 32,
                kernel_size = 7,
                padding = 3,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        example_stats = get_stats(example, self.args).cpu()
                        
        # Process statistics.
        self.stats = nn.Sequential(
            nn.Conv2d(
                in_channels = example_stats.shape[1], 
                out_channels = 32,
                kernel_size = 7,
                padding = 3,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
                
        example_image = self.images(example)
        example_stats = self.stats(example_stats)
        example = torch.cat([example_image, example_stats], dim = 1)
        print("Dis stats and image:", example.shape)

        
        # CNNs shrinking image size.
        self.a = nn.Sequential(
            # 64 by 64
            nn.Dropout2d(p=self.args.dropout),
            CNN_Attention_Blend(
                in_channels = example.shape[1], 
                channels = 32, 
                kernel_size = 7, 
                grow = False,
                shrink = True, 
                paying_attention = False, 
                attention_kernel_size = 5,
                args = default_args),
            nn.Dropout2d(p=self.args.dropout))
        
        example = self.a(example)
        channels_for_pos = 3
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 16, 16) * .5)
        example = add_position_layers(example, self.learned_pos_32, scale = 2)
        print("Dis a:", example.shape)
        
        self.b = nn.Sequential(
            # 32 by 32
            nn.Dropout2d(p=self.args.dropout),
            CNN_Attention_Blend(
                in_channels = example.shape[1], 
                channels = 32, 
                kernel_size = 7, 
                grow = False,
                shrink = True, 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            nn.Dropout2d(p=self.args.dropout),)
            # 16 by 16
            
        example = self.b(example)
        channels_for_pos = 3
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        example = add_position_layers(example, self.learned_pos_16, scale = 2)
        print("Dis b:", example.shape)
        
        self.c = nn.Sequential(
            CNN_Attention_Blend(
                in_channels = example.shape[1], 
                channels = 32, 
                kernel_size = 5, 
                grow = False,
                shrink = True, 
                paying_attention = True, 
                attention_kernel_size = 3,
                args = default_args),
            # 8 by 8
            nn.Dropout2d(p=self.args.dropout),
            CNN_Attention_Blend(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow = False,
                shrink = True, 
                paying_attention = False, 
                attention_kernel_size = 1,
                args = default_args))
            # 4 by 4
                
        example = self.c(example).view(self.args.batch_size, -1)
        print("Dis c:", example.shape)
        
        # Flatten.
        self.d = nn.Sequential(
            nn.Linear(example.shape[-1], self.args.inner_state_size))
        
        example = self.d(example)
        print("Dis d:", example.shape)
        
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
        
        example_mu, example_std = var(example, self.mu, self.std, self.args)
        example = sample(example_mu, example_std)
        print("Finished Discriminator:", example.shape, "\n")
        
        self.apply(init_weights)
        self.to(self.args.device)
        
        

    def forward(self, images):
        batch_size, num_channels, height, width = images.size()
        images = (images * 2) - 1
        images = add_position_layers(images, self.learned_pos_64, scale = 4)
                        
        # Process statistics and images.
        stats = get_stats(images, self.args)
        stats = self.stats(stats)
        images = self.images(images)
        images = torch.cat([images, stats], dim = 1)
    
        # Shrinking and flattening.
        a = self.a(images)
        a = add_position_layers(a, self.learned_pos_32, scale = 2)
        
        b = self.b(a)
        b = add_position_layers(b, self.learned_pos_16, scale = 2)
        
        c = self.c(b)
        d = self.d(c.view(batch_size, -1))
        
        # Apply mean and standard deviation.
        mu, std = var(d, self.mu, self.std, self.args)
        sampled = sample(mu, std)
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
        
    

    