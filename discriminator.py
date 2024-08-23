import torch 
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributions import Normal

from utils import default_args, init_weights, var, sample, generate_2d_sinusoidal_positions



class Discriminator(nn.Module):
    def __init__(self, args = default_args):
        super(Discriminator, self).__init__()
        
        self.args = args
        
        example = torch.zeros(self.args.batch_size, 3 + self.args.pos_channels + 3 + 3 + 3, self.args.image_size, self.args.image_size)
        
        
        self.complete_median = nn.Sequential(
            nn.Conv2d(
                3, 
                3, 
                kernel_size=3, 
                padding=1),  
            nn.LeakyReLU())
        
        self.median = nn.Sequential(
            nn.Conv2d(
                1, 
                3, 
                kernel_size=3, 
                padding=1),  
            nn.LeakyReLU())
        
        self.pixel_median = nn.Sequential(
            nn.Conv2d(
                1, 
                3, 
                kernel_size=3, 
                padding=1),  
            nn.LeakyReLU())
        
        
        
        self.a = nn.Sequential(
            nn.Conv2d(
                3 + self.args.pos_channels + 3 + 3 + 3, 
                32, 
                kernel_size=5, 
                padding=2),  
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                32, 
                32, 
                kernel_size=5, 
                padding=2), 
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                32, 
                32, 
                kernel_size=3, 
                padding=1),  
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        example = self.a(example)
        example = example.view(self.args.batch_size, -1)
        
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

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        x = (x * 2) - 1
        
        complete_median = torch.median(x, dim=0, keepdim=True).values
        complete_median = self.complete_median(complete_median)
        complete_median_tiled = complete_median.repeat(batch_size, 1, 1, 1)
        
        median = torch.median(x, dim=1, keepdim=True).values
        median = self.median(median)
        
        x_reshaped = x.view(x.size(0), x.size(1), -1)
        pixel_median = torch.median(x_reshaped, dim=2, keepdim=True).values
        pixel_median = pixel_median.unsqueeze(-1)
        pixel_median_tiled = pixel_median.repeat(1, 1, 32, 32)
    
        positional_layers = generate_2d_sinusoidal_positions(
            batch_size = x.shape[0], 
            image_size = x.shape[2], 
            d_model = self.args.pos_channels,
            device=self.args.device)
        
        x = torch.cat([x, positional_layers, complete_median_tiled, median, pixel_median_tiled], dim = 1)
        out = self.a(x)
        out = out.view(batch_size, -1)
        
        out = self.b(out)
        
        mu, std = var(out, self.mu, self.std, self.args)
        sampled = sample(mu, std, self.args.device)
        action = torch.tanh(sampled)
        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)  
        
        out = self.c(action)
        out = (out + 1) / 2
        return out, log_prob



if(__name__ == "__main__"):
    dis = Discriminator()
    print("\n\n")
    print(dis)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dis, (16, 3, 32, 32)))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))