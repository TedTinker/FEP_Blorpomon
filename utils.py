#%%

import os

# Your file-location here.
os.chdir(r"C:\Users\Ted\OneDrive\Desktop\FEP_Blorpomon")

from PIL import Image
import datetime 
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import math
import builtins
from math import exp
import imageio
from statistics import log

import torch 
from torch import nn
from torchvision import transforms
from torch.distributions import Normal
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\nDevice: {}.\n\n".format(device))



# Some utilities.
def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)
    
start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time# - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)



# Arguments.
parser = argparse.ArgumentParser()

    # Meta
parser.add_argument("--arg_title",                      type=str,       default = "default",
                    help='If using the cluster, an extensive name for these arguments.') 
parser.add_argument("--arg_name",                       type=str,       default = "default",
                    help='If using the cluster, a name for these arguments.')  
parser.add_argument("--agents",                         type=int,       default = 1,
                    help='If using the cluster, the number of agents trained with these arguments.') 
parser.add_argument("--previous_agents",                type=int,       default = 0,
                    help='If using the cluster, the number of agents before this one.') 
parser.add_argument("--comm",                           type=str,       default = "deigo",
                    help='If using the cluster, name of the cluster in use.') 
parser.add_argument("--init_seed",                      type=float,     default = 777,  # I'm not sure this is working. 
                    help='For consistent randomness.') 
parser.add_argument("--device",                         type=str,       default = device,
                    help='Either cpu or cuda.') 

    # Easy options
parser.add_argument("--epochs",                         type=int,       default = 10000,
                    help='How many epochs for training?') 
parser.add_argument("--batch_size",                     type=int,       default = 64,
                    help='How large are the batches used in epochs?') 
parser.add_argument("--dropout",                        type=int,       default = .01,
                    help='How much dropout for the discriminator?') 
parser.add_argument("--image_size",                     type=int,       default = 64,
                    help='How large are the pictures? (Not used much.)') 
parser.add_argument("--seed_size",                      type=int,       default = 128,
                    help='How large are the seeds used by the generator?') 
parser.add_argument("--inner_state_size",               type=int,       default = 128,
                    help='How large are some linear layers.') 
parser.add_argument('--std_min',                        type=int,       default = exp(-20),
                    help='Minimum value for standard deviation.') 
parser.add_argument('--std_max',                        type=int,       default = exp(2),
                    help='Maximum value for standard deviation.') 
parser.add_argument("--gen_lr",                         type=float,     default = .001,
                    help='Learning rate for generator.') 
parser.add_argument("--dis_lr",                         type=float,     default = .0001,
                    help='Learning rate for discriminator')  
parser.add_argument("--dises",                          type=int,       default = 2,
                    help='How many discriminators?') 
parser.add_argument("--flips",                          type=int,       default = 4,
                    help='How many real images and fake images are swapped?') 
parser.add_argument("--min_real",                       type=float,     default = .7,
                    help='Real images are typically labeled as 1, but it can help to reduce that.') 
parser.add_argument("--max_real",                       type=float,     default = .9,
                    help='Real images are typically labeled as 1, but it can help to reduce that.')  

    # Awesome options
parser.add_argument('--extrinsic',                      type=float,     default = 5,
                    help='Value of extrinsic rewards (generating good pictures).') 
parser.add_argument('--alpha',                          type=float,     default = .05,
                    help='How much generator\'s entropy is rewarded.') 
parser.add_argument('--beta',                           type=float,     default = 1,
                    help='How much generator\'s curiosity is rewarded.') 
parser.add_argument('--dis_alpha',                      type=float,     default = 0,
                    help='How much discriminator\'s entropy is punished.') 
parser.add_argument('--min_dis_std',                    type=float,     default = 0,
                    help='The discriminator\'s goal for standard deviation.') 

    # Presentation options
parser.add_argument("--epochs_per_vid",                 type=int,       default = 25,
                    help='How often are pictures and videos saved?') 
parser.add_argument("--seeds_used",                     type=int,       default = 10,
                    help='When making pictures and videos, how many seeds?') 
parser.add_argument("--seed_duration",                  type=int,       default = 10,
                    help='When making pictures and videos, how many steps transationing from one to another?') 



# Comparing used arguments to default arguments.
try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    
    
        
# Making a title for these arguments.
args_not_in_title = ["arg_title", "init_seed"]
def get_args_title(default_args, args):
    if(args.arg_title[:3] == "___"): return(args.arg_title)
    name = "" ; first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index("arg_name")))
    for arg in arg_list:
        if(arg in args_not_in_title): pass 
        else: 
            default, this_time = getattr(default_args, arg), getattr(args, arg)
            if(this_time == default): pass
            elif(arg == "arg_name"):
                name += "{} (".format(this_time)
            else: 
                if first: first = False
                else: name += ", "
                name += "{}: {}".format(arg, this_time)
    if(name == ""): name = "default" 
    else:           name += ")"
    if(name.endswith(" ()")): name = name[:-3]
    parts = name.split(',')
    name = "" ; line = ""
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): name += line + "\n" ; line = ""
        line += part
        if(i+1 != len(parts)): line += ","
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)



# Use random seed.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(int(args.init_seed))



# Collecting pictures.
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()])

image_files = [f for f in os.listdir("real_images") if os.path.isfile(os.path.join("real_images", f))]
image_files = [f for f in image_files if f != "original.png"]
image_files.sort()
images = []
for file_name in image_files:
    image_path = os.path.join("real_images", file_name)
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_tensor = transform(image)
    images.append(image_tensor)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image_tensor = transform(flipped_image)
    images.append(flipped_image_tensor)
all_images_tensor = torch.stack(images).to(device)



# For batch collection.
def get_random_batch(all_images_tensor = all_images_tensor, batch_size=64):
    num_images = all_images_tensor.size(0)
    indices = random.sample(range(num_images), batch_size)
    batch_tensor = all_images_tensor[indices]
    return batch_tensor

# How to make layers showing positions.
def position_layers(x):
    batch_size, num_channels, height, width = x.size()
    h_grad = torch.linspace(0, 1, steps=width, device=x.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
    v_grad = torch.linspace(0, 1, steps=height, device=x.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    return(h_grad, v_grad)
    


# Make pictures, then make gif transitioning between them.
def show_images_from_tensor(image_tensor, save_path='output_folder', fps=10):
    save_path = f"C:\\Users\\Ted\\OneDrive\\Desktop\\FEP_Blorpomon\\generated_images/{save_path}"
    os.makedirs(save_path, exist_ok=True)

    image_tensor = image_tensor.detach()
    if image_tensor.dim() == 5:
        N, T, C, H, W = image_tensor.shape
        animate = True
    elif image_tensor.dim() == 4:
        N, C, H, W = image_tensor.shape
        T = 1
        animate = False
    else:
        raise ValueError("Unexpected tensor shape")

    frames = []
    frame_index = 1
    for t in range(T):
        for i in range(N):
            img = image_tensor[i, t] if animate else image_tensor[i]
            img = img.permute(1, 2, 0).to("cpu").numpy()

            # Normalize the image to be between 0 and 1
            img = (img - img.min()) / (img.max() - img.min())

            # Convert numpy array to PIL image directly
            pil_image = Image.fromarray((img * 255).astype(np.uint8))  # Assuming image is in [0, 1] range

            # Save the image as a PNG file in the specified folder
            image_filename = os.path.join(save_path, f'{frame_index}.png')
            pil_image.save(image_filename)

            # Append image to frames list for GIF creation
            frames.append(pil_image)
            frame_index += 1

    # Create and save the GIF
    gif_path = os.path.join(save_path, 'animation.gif')
    resized_frames = [frame.resize((frame.width * 20, frame.height * 20), Image.NEAREST) for frame in frames]
    resized_frames[0].save(gif_path, save_all=True, append_images=resized_frames[1:], loop=0, duration=1000//fps)
    
    
    
# Make gifs for epoch-to-epoch progress.
def make_animation(save_dir, image_name='1.png', output_name='animation_1.gif'):
    # Get list of epoch folders sorted by epoch number
    folders = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('epoch_')],
        key=lambda x: int(x.split('_')[1])
    )

    images = []
    for folder in folders:
        path = os.path.join(save_dir, folder, image_name)
        if os.path.exists(path):
            images.append(imageio.imread(path))
    
    output_path = os.path.join(save_dir, output_name)
    imageio.mimsave(output_path, images, fps=5)
    print(f"Saved animation to {output_path}")
    
    
    
# Plotting losses, entropy, curiosity, etc.
def plot_vals(plot_vals_dict, save_path='losses.png', fontsize = 7):
    # Calculate average discriminator losses
    avg_dis_loss_real = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_real"]]
    avg_dis_loss_fake = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_fake"]]
    avg_dis_complexity_loss = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_complexity_loss"]]
        
    # Calculate average discriminator correct rates
    avg_correct_rate_real = [100 * sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_real"]]
    avg_correct_rate_fake = [100 * sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_fake"]]
    
    # Calculate average discriminator mu and std
    avg_mu  = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_mu"]]
    avg_std_fake = [log(sum(epoch)/len(epoch)) for epoch in plot_vals_dict["dis_std_fake"]]
    avg_std_real = [log(sum(epoch)/len(epoch)) for epoch in plot_vals_dict["dis_std_real"]]
        
    # Define epochs
    epochs = range(1, len(plot_vals_dict["gen_loss"]) + 1)
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    
    total_gen_loss = []
    for l, e, c in zip(plot_vals_dict["gen_loss"], plot_vals_dict["gen_entropy_loss"], plot_vals_dict["gen_curiosity_loss"]):
        total_gen_loss.append(l + e + c)
    plt.subplot(2, 3, 1)
    plt.plot(epochs, plot_vals_dict["gen_loss"], 'red', label="Generator Loss", alpha = .8)
    plt.plot(epochs, plot_vals_dict["gen_entropy_loss"], 'green', label="Loss for Entropy", alpha = .8)
    plt.plot(epochs, plot_vals_dict["gen_curiosity_loss"], 'blue', label="Loss for Curiosity", alpha = .8)
    plt.plot(epochs, total_gen_loss, 'black', label="Total", alpha = .8)
    plt.xlabel("Epochs")
    plt.ylabel("Generator Loss")
    plt.ylim(-1, 25)
    plt.title("Generator Losses Over Epochs")
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    
    total_dis_loss = []
    for r, f, c in zip(avg_dis_loss_real, avg_dis_loss_fake, avg_dis_complexity_loss):
        total_dis_loss.append(r + f + c)
    plt.subplot(2, 3, 2)
    plt.plot(epochs, avg_dis_loss_real, 'red', label="Discriminator Loss (real images)", alpha = .8)
    plt.plot(epochs, avg_dis_loss_fake, 'green', label="Discriminator Loss (fake images)", alpha = .8)
    #plt.plot(epochs, avg_dis_complexity_loss, 'blue', label="Loss for Complexity", alpha = .8)
    plt.plot(epochs, total_dis_loss, 'black', label="Total", alpha = .8)
    plt.xlabel("Epochs")
    plt.ylabel("Discriminator Loss")
    plt.ylim(0, 3)
    plt.title("Discriminator Losses Over Epochs")
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    
    # Plot correct rates
    plt.subplot(2, 3, 3)
    plt.plot(epochs, avg_correct_rate_real, 'red', label="Correct Rate (real images)", alpha = .8)
    plt.plot(epochs, avg_correct_rate_fake, 'green', label="Correct Rate (fake images)", alpha = .8)
    plt.xlabel("Epochs")
    plt.ylabel("Correct Rate")
    plt.ylim(0, 100)
    plt.title("Discriminator Correct Rates Over Epochs")
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    
    # Plot generator std
    plt.subplot(2, 3, 4)
    plt.plot(epochs, plot_vals_dict["gen_std"], 'red', label="Generator STD", alpha = .8)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.ylim(0, 1.3)
    plt.title("Generator Standard Deviation")
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    
    # Plot discriminator stda
    plt.subplot(2, 3, 5)
    plt.plot(epochs, avg_std_real, 'red', label="Log Discriminator STD (real images)", alpha = .8)
    plt.plot(epochs, avg_std_fake, 'green', label="Log Discriminator STD (fake images)", alpha = .8)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.ylim(-8, .1)
    plt.title("Discriminator Standard Deviations")
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    
    plt.tight_layout()
    save_path = f"generated_images/{save_path}"
    plt.savefig(save_path)
    plt.close()
    
    

# Quick example.
if(__name__ == "__main__"):
    batch_size = 8
    batch_tensor = get_random_batch(all_images_tensor, batch_size)
    print("Batch shape:", batch_tensor.shape)
    show_images_from_tensor(batch_tensor)
    


# For making smoothly transitioning seeds.
def make_fourier_loop(num_frames, latent_dim, num_frequencies=4):
    t = torch.linspace(0, 2 * math.pi, num_frames, dtype=torch.float32)
    latent_path = torch.zeros(num_frames, latent_dim)
    for k in range(1, num_frequencies + 1):
        a_k = torch.randn(latent_dim) / k
        b_k = torch.randn(latent_dim) / k
        latent_path += torch.sin(k * t[:, None]) * a_k + torch.cos(k * t[:, None]) * b_k
    latent_path = latent_path / latent_path.std()
    return latent_path

def create_interpolated_tensor(args):
    return make_fourier_loop(
        num_frames=args.seeds_used * args.seed_duration,
        latent_dim=args.seed_size,
        num_frequencies=4)



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
                nn.MaxPool2d(
                    kernel_size = 2,
                    stride = 2),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU())
        
        
        
        if(paying_attention):
            self.attention = nn.Sequential(
                SelfAttention(
                    in_channels,
                    attention_kernel_size))
            
            
            
        if(grow_or_shrink in ["none", "shrink"]):
            self.x_out = nn.Sequential(
                nn.Conv2d(
                    in_channels = mid_channels, 
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
                x = F.max_pool2d(input = x, kernel_size = 2, stride = 2)
            attention = self.attention(x)
            x_2 = x_2 + attention
        x = self.x_out(x_2)
        x = self.activations(x)
        return x
    
# %%
