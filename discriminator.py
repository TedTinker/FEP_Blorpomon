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
from utils_for_torch import init_weights, var, sample, Multi_Kernel_CAB, SpaceToDepth, get_stats, add_position_layers, ConstrainedConv2d, rgb_to_circular_hsv



# Let's make a discriminator!
class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to see qualities that layers should have.
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        print("\nStart of Discriminator:", example.shape)
        
        if(self.args.use_hsv):
            example_hsv = rgb_to_circular_hsv(example)
            print("Dis HSV:", example_hsv.shape)
        
        channels_for_pos = 1
        self.learned_pos_64 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        example = add_position_layers(example, self.learned_pos_64, scale = 8)
        example_stats = get_stats(example, True, self.args).cpu()
        
        # Process images.
        self.images = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                grow = False,
                shrink = False, 
                paying_attention = False, 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout))
                                
        # Process statistics.
        self.stats = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example_stats.shape, 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                grow = False,
                shrink = False, 
                paying_attention = False, 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout))
                
        example_image = self.images(example)
        example_stats = self.stats(example_stats)
        example = torch.cat([example_image, example_stats], dim = 1)
        print("Dis image and stats:", example.shape)
        
        # Process images HSV.
        if(self.args.use_hsv):
            example_hsv = add_position_layers(example_hsv, self.learned_pos_64, scale = 8)
            example_hsv_stats = get_stats(example_hsv, False, self.args).cpu()
            
            self.hsv = nn.Sequential(
                Multi_Kernel_CAB(
                    in_shape = example_hsv.shape, 
                    out_channels = [16, 8, 8], 
                    kernel_sizes = [3, 5, 7], 
                    grow = False,
                    shrink = False, 
                    paying_attention = False, 
                    args = self.args),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Dropout2d(p=self.args.dropout))
            
            self.hsv_stats = nn.Sequential(
                Multi_Kernel_CAB(
                    in_shape = example_hsv_stats.shape, 
                    out_channels = [16, 8, 8], 
                    kernel_sizes = [3, 5, 7], 
                    grow = False,
                    shrink = False, 
                    paying_attention = False, 
                    args = self.args),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Dropout2d(p=self.args.dropout))
            
            example_hsv = self.hsv(example_hsv)
            example_hsv_stats = self.hsv_stats(example_hsv_stats)
            example = torch.cat([example, example_hsv, example_hsv_stats], dim = 1)
            print("Dis image, stats, HSV, and HSV stats:", example.shape)

        # CNNs shrinking image size.
        self.a = nn.Sequential(
            # 64 by 64
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                grow = False,
                shrink = True, 
                paying_attention = False, 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout))
        
        example = self.a(example)
        channels_for_pos = 1
        self.learned_pos_32 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        example = add_position_layers(example, self.learned_pos_32, scale = 4)
        print("Dis a:", example.shape)
        
        self.b = nn.Sequential(
            # 32 by 32
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [16, 8, 8], 
                kernel_sizes = [3, 5, 7], 
                grow = False,
                shrink = True, 
                paying_attention = False, 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout))
            # 16 by 16
            
        example = self.b(example)
        channels_for_pos = 1
        self.learned_pos_16 = nn.Parameter(torch.ones(1, channels_for_pos, 8, 8) * .5)
        example = add_position_layers(example, self.learned_pos_16, scale = 2)
        print("Dis b:", example.shape)
        
        self.c = nn.Sequential(
            Multi_Kernel_CAB(
                in_shape = example.shape, 
                out_channels = [24, 8], 
                kernel_sizes = [3, 5],  
                grow = False,
                shrink = True, 
                paying_attention = False, 
                args = self.args),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout),
            # 8 by 8
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 8,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            SpaceToDepth(block_size=2),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=self.args.dropout))
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
        
        if(self.args.use_hsv):
            hsv = rgb_to_circular_hsv(images)
            hsv = add_position_layers(hsv, self.learned_pos_64, scale = 8)
            
        images = (images * 2) - 1
        images = add_position_layers(images, self.learned_pos_64, scale = 8)
                        
        # Process statistics and images.
        stats = get_stats(images, True, self.args)
        stats = self.stats(stats)
        images = self.images(images)
        images = torch.cat([images, stats], dim = 1)
        
        if(self.args.use_hsv):
            hsv_stats = get_stats(hsv, False, self.args)
            hsv_stats = self.hsv_stats(hsv_stats)
            hsv = self.hsv(hsv)
            images = torch.cat([images, hsv, hsv_stats], dim = 1)
    
        # Shrinking and flattening.
        a = self.a(images)
        a = add_position_layers(a, self.learned_pos_32, scale = 4)
        
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
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
    

    