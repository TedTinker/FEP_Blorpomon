#%%

# To do:
# Try plotting the positional channels, to see if they make sense.
# Try generative larger images and cropping.
# seed-sequence-maker not perfect
# Should black and white be two separate channels, or one?

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
from kornia.color import rgb_to_hsv 

if(not "FEP_Blorpomon" in [os.getcwd().split("/")[-1], os.getcwd().split("\\")[-1]]): 
                           os.chdir("FEP_Blorpomon")

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
parser.add_argument("--init_seed",                      type=float,     default = 777)
parser.add_argument("--agents",                         type=int,       default = 1) 
parser.add_argument("--previous_agents",                type=int,       default = 0)
parser.add_argument("--device",                         type=str,       default = device)

    # Easy options
parser.add_argument("--epochs",                         type=int,       default = 10000) 
parser.add_argument("--batch_size",                     type=int,       default = 64) 

    # Harder options
parser.add_argument("--image_size",                     type=int,       default = 64)
parser.add_argument("--seed_size",                      type=int,       default = 128)
parser.add_argument("--inner_state_size",               type=int,       default = 128)
parser.add_argument("--median_size",                    type=int,       default = 128)
parser.add_argument('--std_min',                        type=int,       default = exp(-20))
parser.add_argument('--std_max',                        type=int,       default = exp(2))
parser.add_argument("--gen_lr",                         type=float,     default = .001) 
parser.add_argument("--dis_lr",                         type=float,     default = .001) 
parser.add_argument("--dises",                          type=int,       default = 2) 
parser.add_argument("--flips",                          type=int,       default = 4) 
parser.add_argument("--min_real",                       type=float,     default = .7) 
parser.add_argument("--max_real",                       type=float,     default = .9) 

    # Awesome options
parser.add_argument('--extrinsic',                      type=float,     default = 1)
parser.add_argument('--alpha',                          type=float,     default = 1)
parser.add_argument('--beta',                           type=float,     default = 1)
parser.add_argument('--dis_alpha',                      type=float,     default = 1)
parser.add_argument('--pos_channels',                   type=int,       default = 0)

    # Presentation options
parser.add_argument("--epochs_per_vid",                 type=int,       default = 10)
parser.add_argument("--seeds_used",                     type=int,       default = 6)
parser.add_argument("--seed_duration",                  type=int,       default = 10)



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

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
class Ted_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes = [(1,1),(3,3),(5,5)], stride = 1):
        super(Ted_Conv2d, self).__init__()
        
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
    
    
    
def rgb_to_circular_hsv(rgb):
    hsv_image = rgb_to_hsv(rgb) 
    hue = hsv_image[:, 0, :, :]
    hue_sin = (torch.sin(hue) + 1) / 2
    hue_cos = (torch.cos(hue) + 1) / 2
    hsv_circular = torch.stack([hue_sin, hue_cos, hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]], dim=1)
    return hsv_circular



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




def show_images_from_tensor(image_tensor, save_path='output_folder', fps=10):
    # Ensure the save path is a directory
    save_path = f"generated_images/{save_path}"
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


if __name__ == "__main__":
    show_images_from_tensor(batch_tensor)
    
    
    
def plot_vals(plot_vals_dict, save_path='losses.png'):
    # Calculate average discriminator losses
    avg_dis_loss_real = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_real"]]
    avg_dis_loss_fake = [sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_losses_fake"]]
    
    # Calculate average correct rates
    avg_correct_rate_real = [100 * sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_real"]]
    avg_correct_rate_fake = [100 * sum(epoch)/len(epoch) for epoch in plot_vals_dict["dis_correct_rate_fake"]]
    
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
    plt.ylim(0, 100)
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
