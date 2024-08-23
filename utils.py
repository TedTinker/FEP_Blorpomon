#%%

# To do:
# Make it work on deigo
# seed-sequence-maker not perfect
# Instead of medians, also use quantiles etc

from PIL import Image
import datetime 
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse 
import math
import builtins
from math import exp, log
import scipy.stats as stats

import torch 
from torch import nn
from torchvision import transforms
from torch.distributions import Normal

if(os.getcwd().split("/")[-1] != "FEP_Blorpomon"): os.chdir("FEP_Blorpomon")
print(os.getcwd())

def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\nDevice: {}.\n\n".format(device))



parser = argparse.ArgumentParser()

    # Meta
parser.add_argument("--arg_title",                      type=str,           default = "default") 
parser.add_argument("--arg_name",                       type=str,           default = "default") 
parser.add_argument("--comm",                           type=str,           default = "deigo")
parser.add_argument("--init_seed",          type=float,     default = 777)
parser.add_argument("--agents",             type=int,       default = 1) 
parser.add_argument("--previous_agents",    type=int,       default = 0)
parser.add_argument("--device",             type=str,       default = device)

    # Easy options
parser.add_argument("--epochs",             type=int,       default = 10000) 
parser.add_argument("--batch_size",         type=int,       default = 64) 

    # Harder options
parser.add_argument("--image_size",         type=int,       default = 32)
parser.add_argument("--seed_size",          type=int,       default = 128)
parser.add_argument("--inner_state_size",   type=int,       default = 128)
parser.add_argument("--median_size",        type=int,       default = 128)
parser.add_argument('--std_min',            type=int,       default = exp(-20))
parser.add_argument('--std_max',            type=int,       default = exp(2))
parser.add_argument("--gen_lr",             type=float,     default = .001) 
parser.add_argument("--dis_lr",             type=float,     default = .001) 
parser.add_argument("--dises",              type=int,       default = 2) 

    # Awesome options
parser.add_argument('--extrinsic',          type=float,     default = 1)
parser.add_argument('--alpha',              type=float,     default = 1)
parser.add_argument('--beta',               type=float,     default = 1)
parser.add_argument('--dis_alpha',          type=float,     default = 1)
parser.add_argument('--pos_channels',       type=int,       default = 4)

    # Presentation options
parser.add_argument("--epochs_per_vid",     type=int,       default = 100)
parser.add_argument("--seeds_used",         type=int,       default = 6)
parser.add_argument("--seed_duration",      type=int,       default = 10)



try:
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
except:
    import sys ; sys.argv=[''] ; del sys           
    default_args = parser.parse_args([])
    try:    args    = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    
    
    
#for arg_set in [default_args, args]:
    #arg_set.sensors_shape = num_sensors
        
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


    
def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

def var(x, mu_func, std_func, args):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = args.std_min, max = args.std_max)
    return(mu, std)

def sample(mu, std, device):
    e = Normal(0, 1).sample(std.shape).to(device)
    return(mu + e * std)



if(__name__ == "__main__"):
    image = Image.open(f'real_images/original.png')
    width, height = image.size
    tile_num = 0
    for y in range(0, height, 64):
        for x in range(0, width, 64):
            if tile_num not in [0] + [i for i in range(202, 230)] + [i for i in range(760,768)]:
                box = (x, y, x + 64, y + 64)
                tile = image.crop(box)
                tile.save(os.path.join(f'real_images/{str(tile_num).zfill(3)}.png'))
            tile_num += 1
    print(f"Total {tile_num} tiles created.")



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



def get_random_batch(all_images_tensor = all_images_tensor, batch_size=64):
    num_images = all_images_tensor.size(0)
    indices = random.sample(range(num_images), batch_size)
    batch_tensor = all_images_tensor[indices]
    return batch_tensor

if(__name__ == "__main__"):
    batch_size = 8
    batch_tensor = get_random_batch(all_images_tensor, batch_size)
    print(batch_tensor.shape)



def show_images_from_tensor(image_tensor, save_path='animation.gif', fps=10):
    image_tensor = image_tensor.detach()
    if image_tensor.dim() == 5:
        N, T, C, H, W = image_tensor.shape
        animate = True
    if image_tensor.dim() == 4:
        N, C, H, W = image_tensor.shape
        T = 1
        animate = False
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, N, figsize=(N * (W / 10), H / 10), dpi=100)
        if N == 1:
            axes = [axes]
        for i in range(N):
            img = image_tensor[i, t] if animate else image_tensor[i]
            img = img.permute(1, 2, 0).to("cpu").numpy() 
            axes[i].imshow(img)
            axes[i].axis('off')
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pil_image = Image.fromarray(image_array)
        frames.append(pil_image)
        plt.close(fig) 
    save_path = f"generated_images/{save_path}"
    frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=1000//fps)

if(__name__ == "__main__"):
    show_images_from_tensor(batch_tensor)
    
    
    
def plot_vals(plot_vals_dict, save_path='losses.png'):
    # Calculate average discriminator losses
    avg_dis_loss_real = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_real"]]
    avg_dis_loss_fake = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_fake"]]
    
    # Calculate average correct rates
    avg_correct_rate_real = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_real"]]
    avg_correct_rate_fake = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_fake"]]
    
    # Define epochs
    epochs = range(1, len(plot_vals_dict["gen_loss"]) + 1)
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, plot_vals_dict["gen_loss"], 'b-', label="Generator Loss")
    plt.plot(epochs, avg_dis_loss_real, 'g-', label="Discriminator Loss (Real)")
    plt.plot(epochs, avg_dis_loss_fake, 'r-', label="Discriminator Loss (Fake)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Losses Over Epochs")
    plt.legend()
    plt.grid(True)
    
    # Plot correct rates
    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_correct_rate_real, 'g-', label="Correct Rate (Real)")
    plt.plot(epochs, avg_correct_rate_fake, 'r-', label="Correct Rate (Fake)")
    plt.xlabel("Epochs")
    plt.ylabel("Correct Rate")
    plt.title("Discriminator Correct Rates Over Epochs")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = f"generated_images/{save_path}"
    plt.savefig(save_path)
    plt.close()



def slerp(val, low, high):
    """
    Spherical linear interpolation (slerp) between two points on a unit sphere.
    """
    omega = torch.acos(torch.clamp(torch.dot(low / torch.norm(low), high / torch.norm(high)), -1, 1))
    sin_omega = torch.sin(omega)
    if sin_omega == 0:
        return (1.0 - val) * low + val * high  # linear interpolation as fallback
    return (torch.sin((1.0 - val) * omega) / sin_omega) * low + (torch.sin(val * omega) / sin_omega) * high

def create_interpolated_tensor(args):
    seeds = [torch.randn(args.seed_size) for _ in range(args.seeds_used)]
    result = torch.zeros(args.seeds_used * args.seed_duration, args.seed_size)
    for i in range(args.seeds_used):
        current_seed = seeds[i]
        next_seed = seeds[(i + 1) % args.seeds_used]
        for j in range(args.seed_duration):
            alpha = j / args.seed_duration
            interpolated = slerp(alpha, current_seed, next_seed)
            result[i * args.seed_duration + j] = interpolated
    return result



def generate_2d_sinusoidal_positions(batch_size, image_size, d_model=2, device='cpu'):
    assert d_model % 2 == 0, "d_model should be even."
    x = torch.arange(image_size, dtype=torch.float32, device=device).unsqueeze(0).expand(image_size, image_size)
    y = torch.arange(image_size, dtype=torch.float32, device=device).unsqueeze(1).expand(image_size, image_size)
    x = x.unsqueeze(2).tile((1, 1, d_model // 2))
    y = y.unsqueeze(2).tile((1, 1, d_model // 2))

    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-log(10000.0) / (d_model // 2)))
    div_term = torch.tile(div_term.unsqueeze(0).unsqueeze(0), (image_size, image_size, 1))
    
    pe = torch.zeros(image_size, image_size, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(x * div_term) + torch.sin(y * div_term)
    pe[:, :, 1::2] = torch.cos(x * div_term) + torch.cos(y * div_term)
    pe = torch.tile(pe.unsqueeze(0), (batch_size, 1, 1, 1))
    pe = pe.permute(0, 3, 1, 2) / 2
    return pe



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
# %%
