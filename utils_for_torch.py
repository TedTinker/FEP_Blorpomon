import math

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from utils import default_args




# For starting neural networks.
def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    except: pass

# How to use mean and standard deviation layers.
def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

# How to sample from probability distributions.
def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)



# For making smoothly transitioning seeds.
def make_fourier_loop(num_frames, latent_dim, num_frequencies=4, generator=None):
    t = torch.linspace(0, 2 * math.pi, num_frames, dtype=torch.float32)
    latent_path = torch.zeros(num_frames, latent_dim)
    for k in range(1, num_frequencies + 1):
        a_k = torch.randn(latent_dim, generator=generator) / k
        b_k = torch.randn(latent_dim, generator=generator) / k
        latent_path += torch.sin(k * t[:, None]) * a_k + torch.cos(k * t[:, None]) * b_k
    latent_path = latent_path / latent_path.std()
    return latent_path

def create_interpolated_tensor(args):
    g = torch.Generator()
    g.manual_seed(int(args.init_seed))
    return make_fourier_loop(
        num_frames=args.seeds_used * args.seed_duration,
        latent_dim=args.seed_size,
        num_frequencies=4,
        generator=g)



# Pixel shuffling.
def space_to_depth(x, r):
    B, C, H, W = x.shape
    assert H % r == 0 and W % r == 0, "H and W must be divisible by r"
    x = x.view(B, C, H // r, r, W // r, r)  # Reshape to chunk spatial dims
    x = x.permute(0, 1, 3, 5, 2, 4)         # Bring r chunks into channel dim
    x = x.reshape(B, C * r * r, H // r, W // r)
    return x

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        x = space_to_depth(x, 2)
        return x



# CNN with capping.
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
    
    
# Multi-Kernel CNN (MKC).
class Multi_Kernel_CNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [(1,1),(3,3),(5,5)], stride = 1):
        super(Multi_Kernel_CNN, self).__init__()
        
        self.Conv2ds = nn.ModuleList()
        for kernel, out_channel in zip(kernel_sizes, out_channels):
            if(type(kernel) == int): 
                kernel = (kernel, kernel)
            padding = ((kernel[0]-1)//2, (kernel[1]-1)//2)
            layer = nn.Sequential(
                ConstrainedConv2d(
                    in_channels = in_channels,
                    out_channels = out_channel,
                    kernel_size = kernel,
                    padding = padding,
                    padding_mode = "reflect",
                    stride = stride))
            self.Conv2ds.append(layer)
                
    def forward(self, x):
        y = []
        for Conv2d in self.Conv2ds: y.append(Conv2d(x)) 
        return(torch.cat(y, dim = -3))



# Attention layers.
class SelfAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 1):
        super().__init__()
        padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
        self.query = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = in_channels // 8, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.key   = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = in_channels // 8, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.value = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = in_channels, 
            kernel_size = kernel_size, 
            padding = padding_size, 
            padding_mode = "reflect")
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # B x HW x C'
        proj_key   = self.key(x).view(B, -1, H * W)                      # B x C' x HW
        energy     = torch.bmm(proj_query, proj_key)                    # B x HW x HW
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)                   # B x C x HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))        # B x C x HW
        out = out.view(B, C, H, W)
        return self.gamma * out + x
    
    
    
# My personal kind of layer. Allows growing, shrinking, and attention.
class CNN_Attention_Blend(nn.Module):
    def __init__(self, 
                 in_channels = 32, 
                 channels = 32, 
                 kernel_size = 3, 
                 grow = False,
                 shrink = False,
                 paying_attention = False, 
                 attention_kernel_size = 1, 
                 activations = True, 
                 args = default_args):
        super(CNN_Attention_Blend, self).__init__()
        
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        mid_channels = channels
        if(self.shrink and self.paying_attention):
            mid_channels = in_channels
        if(self.grow or (not self.grow and not self.shrink)):
            mid_channels = in_channels
        
        padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
        
        
        if(self.grow):
            self.x_in = nn.Sequential(
                nn.Conv2d(
                    in_channels = channels,
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.Upsample(
                    scale_factor = 2,
                    mode = "bilinear",
                    align_corners = True),
                nn.BatchNorm2d(mid_channels * 4),
                nn.LeakyReLU())
        
        if(self.grow or (not self.grow and not self.shrink)):
            self.x_in = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU())
            
        if(self.shrink):
            self.x_in = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                SpaceToDepth(block_size=2),  
                nn.BatchNorm2d(mid_channels * 4),
                nn.LeakyReLU())
        
        
        
        if(paying_attention):
            self.attention = nn.Sequential(
                SelfAttention(
                    in_channels * (4 if self.shrink else 1),
                    attention_kernel_size))
            
            
            
        if(self.shrink or (not self.grow and not self.shrink)):
            self.x_out = nn.Sequential(
                nn.Conv2d(
                    in_channels = mid_channels * (4 if self.shrink else 1), 
                    out_channels = channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"))
            
        if(self.grow):
            self.x_out = nn.Sequential(
                nn.Conv2d(
                    in_channels = mid_channels,
                    out_channels = channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.Upsample(
                    scale_factor = 2,
                    mode = "bilinear",
                    align_corners = True))
            
        if(activations):
            self.activations = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.LeakyReLU())
        else:
            self.activations = nn.Sequential()
    
    def forward(self, x):
        x_2 = self.x_in(x)
        if(self.paying_attention):
            if(self.shrink):
                x = space_to_depth(x, 2)
            attention = self.attention(x)
            x_2 = x_2 + attention
        x = self.x_out(x_2)
        x = self.activations(x)
        return x