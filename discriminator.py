import torch 
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributions import Normal
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from utils import get_random_batch, default_args, init_weights, var, sample, generate_2d_sinusoidal_positions, rgb_to_circular_hsv, Ted_Conv2d



def display_tensors_as_images(tensors):
    cleaned = []
    for t in tensors:
        t = t.cpu().detach().numpy()
        if(t.shape[1] != 1):
            t = t.transpose(0, 2, 3, 1)  # Convert from (B, C, H, W) to (B, H, W, C)
        else:
            t = t.squeeze(1)  # Remove the channel dimension to get (B, H, W)
        cleaned.append(t)

    fig, axs = plt.subplots(len(cleaned[0]), len(cleaned), figsize=(5 * len(cleaned), 5 * len(cleaned[0])))
    
    names = [" "] * len(cleaned)
    
    for i, (c, n) in enumerate(zip(cleaned, names)):
        axs[0,i].set_title(n)
        for j, r in enumerate(c):
            if(len(r.shape) == 3):
                axs[j,i].imshow(r)
            else:
                axs[j,i].imshow(r, cmap='gray')
            #axs[j,i].axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    img = Image.open(buf)
    img.show()



quantiles = [0.05, 0.5, 0.95]

def get_stats(x, args, display = False):
    batch_size, num_channels, height, width = x.size()
    to_cat = []
    
    if(args.pos_channels != 0):
        positional_layers = generate_2d_sinusoidal_positions(
            batch_size = x.shape[0], 
            image_size = x.shape[2], 
            d_model = args.pos_channels,
            device=args.device)
        to_cat.append(positional_layers)
    
    

    batch_quantiles = [torch.quantile(x, q, dim=0, keepdim=True) for q in quantiles] # (1, channels, width, height)
    batch_quantiles_tiled = [q.repeat(batch_size, 1, 1, 1) for q in batch_quantiles]
    to_cat.extend(batch_quantiles_tiled)
    
    x_reshaped = x.view(x.size(0), x.size(1), -1)
    pixel_quantiles = [torch.quantile(x_reshaped, q, dim=2, keepdim=True) for q in quantiles] # (batch, channels, 1)
    pixel_quantiles_tiled = [q.unsqueeze(-1).repeat(1, 1, height, width) for q in pixel_quantiles]
    to_cat.extend(pixel_quantiles_tiled)
    
    
    
    batch_std = torch.std(x, dim=0, keepdim=True) # (1, channels, width, height)
    batch_std_tiled = batch_std.repeat(batch_size, 1, 1, 1)
    to_cat.append(batch_std_tiled)

    pixel_std = torch.std(x_reshaped, dim=2, keepdim=True) # (batch, channels, 1)
    pixel_std = pixel_std.unsqueeze(-1)
    pixel_std_tiled = pixel_std.repeat(1, 1, height, width)
    to_cat.append(pixel_std_tiled)
        
    
    
    max_rgb, _ = x.max(dim=1, keepdim=True)
    min_rgb, _ = x.min(dim=1, keepdim=True)
    delta = max_rgb - min_rgb
    v = max_rgb
    s = delta / (max_rgb + 1e-7)  # Add a small constant to avoid division by zero
    
    brightness_threshold_white = 0.9
    brightness_threshold_black = 0.9
    saturation_threshold_white = 0.1  # Low saturation to consider color close to grayscale for white
    saturation_threshold_black = 0.1  # Low saturation to consider color close to grayscale for black
    w = torch.where((v >= brightness_threshold_white) & (s <= saturation_threshold_white), torch.ones_like(v), torch.zeros_like(v))
    b = torch.where((v <= brightness_threshold_black) & (s <= saturation_threshold_black), -torch.ones_like(v), torch.zeros_like(v))
    wb = w + b
    #to_cat.append(wb) # These help the discriminator SO MUCH.
                
    batch_wb_mean = torch.mean(wb, dim=0, keepdim=True) # (1, channels, height, width)
    batch_wb_mean_tiled = batch_wb_mean.repeat(args.batch_size, 1, 1, 1)
    #to_cat.append(batch_wb_mean_tiled)
    
    if(display):
        print("\n")
        for t in [x, v, w, b, wb]:
            print(f"Shape: {t.shape}, Min: {(t.min() + 1) / 2}, Max: {(t.max() + 1) / 2}")
        how_many = 10
        display_tensors_as_images([
            (x[:how_many] + 1) / 2, 
            (v[:how_many] + 1) / 2,
            (w[:how_many] + 1) / 2,
            (b[:how_many] + 1) / 2,
            (wb[:how_many] + 1) / 2,
            (batch_wb_mean_tiled[:how_many] + 1) / 2])
            
            
            
    statistics = torch.cat(to_cat, dim = 1)
    return(statistics)



class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        
        stats = get_stats(example, self.args)
        stat_channels = stats.shape[1]
                        
        self.stats = nn.Sequential(
            Ted_Conv2d(
                stat_channels,
                [32 // 4] * 4,
                kernel_sizes = [1, 3, 5, 7]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        stats = self.stats(stats)
        
        self.images = nn.Sequential(
            Ted_Conv2d(
                3,
                [32 // 4] * 4,
                kernel_sizes = [3, 5, 5, 7]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        example = self.images(example)
        
        example = torch.cat([example, stats], dim = 1)
        
        self.a = nn.Sequential(
            
            # 32 by 32
            
            Ted_Conv2d(
                example.shape[1],
                [32 // 4] * 4,
                kernel_sizes = [3, 5, 5, 7]),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 16 by 16
            
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
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 8 by 8
            
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
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 4 by 4
            
            Ted_Conv2d(
                32,
                [32 // 4] * 4,
                kernel_sizes = [1, 1, 3, 3]),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
                
        example = self.a(example).view(self.args.batch_size, -1)
        
        self.b = nn.Sequential(
            nn.Linear(example.shape[-1], self.args.inner_state_size))
        
        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.inner_state_size, 
                out_features =  self.args.inner_state_size))
        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.inner_state_size, 
                out_features =  self.args.inner_state_size),
            nn.Softplus())
        
        self.c = nn.Sequential(
            nn.Linear(
                self.args.inner_state_size, 
                1),
            nn.Tanh())
        
        self.apply(init_weights)
        self.to(self.args.device)

    def forward(self, images, display = False):
        batch_size, num_channels, height, width = images.size()
        #hsv = rgb_to_circular_hsv(images)
        #images = torch.cat([images, hsv], dim = 1)
        images = (images * 2) - 1
                
        stats = get_stats(images, self.args, display)
        stats = self.stats(stats)
        images = self.images(images)
        images = torch.cat([images, stats], dim = 1)
        a = self.a(images).view(batch_size, -1)
        b = self.b(a)
        
        mu, std = var(b, self.mu, self.std, self.args)
        sampled = sample(mu, std, self.args.device)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)  
        
        out = self.c(action)
        out = (out + 1) / 2
        return out, log_prob



if(__name__ == "__main__"):
    args = default_args
    dis = Discriminator(args)
    print("\n\n")
    print(dis)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dis, (args.batch_size, 3, 32, 32)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    def generate_white_to_black_transition(batch_size, height, width, device='cpu'):
        transition = torch.linspace(1, 0, width).unsqueeze(0).unsqueeze(0).repeat(batch_size, height, 1)
        transition_tensor = transition.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        return transition_tensor
    
    #dis(generate_white_to_black_transition(args.batch_size, 32, 32, args.device), display = True)    
    dis(get_random_batch(batch_size = args.batch_size), display = True)
    
    