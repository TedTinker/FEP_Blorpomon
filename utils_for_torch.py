import math

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from utils import default_args



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



# How to make layers showing positions.
def position_layers(x):
    batch_size, num_channels, height, width = x.size()
    h_grad = torch.linspace(0, 1, steps=width, device=x.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
    v_grad = torch.linspace(0, 1, steps=height, device=x.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    return(h_grad, v_grad)



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
class My_Layer(nn.Module):
    def __init__(self, 
                 in_channels = 32, 
                 channels = 32, 
                 kernel_size = 3, 
                 grow_or_shrink = "none", 
                 paying_attention = False, 
                 attention_kernel_size = 1, 
                 activations = True, 
                 args = default_args):
        super(My_Layer, self).__init__()
        
        self.args = args
        self.grow_or_shrink = grow_or_shrink
        self.paying_attention = paying_attention
        
        mid_channels = channels
        if(grow_or_shrink == "shrink" and paying_attention):
            mid_channels = in_channels
        if(grow_or_shrink in ["none", "grow"]):
            mid_channels = in_channels
        
        padding_size = ((kernel_size-1)//2, (kernel_size-1)//2)
        
        if(grow_or_shrink in ["none", "grow"]):
            self.x_in = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU())
            
        if(grow_or_shrink == "shrink"):
            self.x_in = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"),
                #nn.AvgPool2d(
                #    kernel_size = 2,
                #    stride = 2),
                SpaceToDepth(block_size=2),  
                nn.BatchNorm2d(mid_channels * 4),
                nn.LeakyReLU())
        
        
        
        if(paying_attention):
            self.attention = nn.Sequential(
                SelfAttention(
                    in_channels * (4 if grow_or_shrink == "shrink" else 1),
                    attention_kernel_size))
            
            
            
        if(grow_or_shrink in ["none", "shrink"]):
            self.x_out = nn.Sequential(
                nn.Conv2d(
                    in_channels = mid_channels * (4 if grow_or_shrink == "shrink" else 1), 
                    out_channels = channels,
                    kernel_size = kernel_size,
                    padding = padding_size,
                    padding_mode = "reflect"))
            
        if(grow_or_shrink == "grow"):
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
            if(self.grow_or_shrink == "shrink"):
                #x = F.max_pool2d(input = x, kernel_size = 2, stride = 2)
                x = space_to_depth(x, 2)
            attention = self.attention(x)
            x_2 = x_2 + attention
        x = self.x_out(x_2)
        x = self.activations(x)
        return x