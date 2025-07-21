import os 

os.chdir(r"C:\Users\Ted\Desktop\FEP_Blorpomon")

import torch 
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from utils import get_random_batch, default_args, init_weights, var, sample, My_Layer, position_layers



# A way to check out some statistics we process.
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



# Collecting statistics from batch.
# Some of these statistics are way too helpful for the discriminator.
quantiles = [0.05, .25, .5, .75, 0.95]

def get_stats(x, args, display = False):
    batch_size, num_channels, height, width = x.size()
    x_flat = x.view(x.size(0), x.size(1), -1)  # (batch, channels, height * width)
    to_cat = []
    
    h_grad, v_grad = position_layers(x)
    to_cat.extend([h_grad, v_grad])

    batch_quantiles = [torch.quantile(x, q, dim=0, keepdim=True) for q in quantiles] # (1, channels, width, height)
    batch_quantiles_tiled = [q.repeat(batch_size, 1, 1, 1) for q in batch_quantiles]
    to_cat.extend(batch_quantiles_tiled)
    
    #per_sample_quantiles = [torch.quantile(x_flat, q, dim=2, keepdim=True) for q in quantiles]  # shape: (batch, channels, 1)
    #per_sample_quantiles_tiled = [q.unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3)) for q in per_sample_quantiles]
    #to_cat += per_sample_quantiles_tiled
    
    x_reshaped = x.view(x.size(0), x.size(1), -1)
    pixel_quantiles = [torch.quantile(x_reshaped, q, dim=2, keepdim=True) for q in quantiles] # (batch, channels, 1)
    pixel_quantiles_tiled = [q.unsqueeze(-1).repeat(1, 1, height, width) for q in pixel_quantiles]
    to_cat.extend(pixel_quantiles_tiled)
    
    batch_std = torch.std(x, dim=0, keepdim=True) # (1, channels, width, height)
    batch_std_tiled = batch_std.repeat(batch_size, 1, 1, 1)
    to_cat.append(batch_std_tiled)
    
    #per_sample_std = torch.std(x, dim=(2, 3), keepdim=True)
    #per_sample_std_tiled = per_sample_std.repeat(1, 1, height, width)
    #to_cat.append(per_sample_std_tiled)

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
    #to_cat.append(w)
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
            
    to_cat = [stat.to(args.device) for stat in to_cat]
    statistics = torch.cat(to_cat, dim = 1)
    return(statistics)



# Let's make a discriminator!
class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        # This is my kludgey way to get the number of channels and such layers should have.
        example = torch.zeros(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
        
        stats = get_stats(example, self.args).cpu()
        stat_channels = stats.shape[1]
                        
        # Process statistics.
        self.stats = nn.Sequential(
            nn.Conv2d(
                in_channels = stat_channels, 
                out_channels = 32,
                kernel_size = 5,
                padding = 1,
                padding_mode = "reflect"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        stats = self.stats(stats)
        
        # Process images.
        self.images = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 32,
                kernel_size = 5,
                padding = 1,
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
                kernel_size = 3, 
                grow_or_shrink = "shrink", 
                paying_attention = False, 
                attention_kernel_size = 5,
                args = default_args),
            # 32 by 32
            nn.Dropout2d(p=self.args.dropout),
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "shrink", 
                paying_attention = False, 
                attention_kernel_size = 5,
                args = default_args),
            # 16 by 16
            nn.Dropout2d(p=self.args.dropout),
            My_Layer(
                in_channels = 32, 
                channels = 32, 
                kernel_size = 3, 
                grow_or_shrink = "shrink", 
                paying_attention = False, 
                attention_kernel_size = 1,
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
                
        example = self.a(example).view(self.args.batch_size, -1)
        
        # Flatten.
        self.b = nn.Sequential(
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

    def forward(self, images, display = False):
        batch_size, num_channels, height, width = images.size()
        images = (images * 2) - 1
                        
        # Process statistics and images.
        stats = get_stats(images, self.args, display)
        stats = self.stats(stats)
        images = self.images(images)
        images = torch.cat([images, stats], dim = 1)
                        
        # Shrinking and flattening.
        a = self.a(images).view(batch_size, -1)
        b = self.b(a)
        
        # Apply mean and standard deviation.
        mu, std = var(b, self.mu, self.std, self.args)
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
    
    dis(get_random_batch(batch_size = args.batch_size), display = True)
    
    

    