#%% 
import os

os.chdir(r"C:\Users\Ted\Desktop\FEP_Blorpomon")

import torch 
from torch.optim import Adam
import torch.nn.functional as F

from utils import default_args, get_random_batch, create_interpolated_tensor, show_images_from_tensor, plot_vals, print, duration, make_animation
from generator import Generator
from discriminator import Discriminator



# Let's put all this together!
class GAN:
    def __init__(self, args = default_args):
        self.args = args
        
        # Folder to save in.
        folder_name = "generated_images/" + str(self.args.arg_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), args.gen_lr)
        
        self.dis_list = [Discriminator() for i in range(self.args.dises)]
        self.dis_opts = [Adam(dis.parameters(), args.dis_lr) for dis in self.dis_list]
        
        # These are seeds for consistent pictures and videos every so many epochs.
        self.seeds = create_interpolated_tensor(self.args).to(self.args.device)
        
        # Dictionary for plotting loss-information, etc.
        self.plot_vals_dict = {
            "dis_correct_rate_real" : [[]],
            "dis_correct_rate_fake" : [[]],
            "dis_losses_real" : [[]],
            "dis_losses_fake" : [[]],
            "dis_complexity_loss" : [[]],
            "dis_mu" : [[]],
            "dis_std" : [[]],
            "gen_loss" : [],
            "gen_entropy_loss" : [],
            "gen_curiosity_loss" : [],
            "gen_mu" : [],
            "gen_std" : []
            }
        
        self.epochs = 1
        
    # One step of training.
    def epoch(self):
        
        if(self.epochs % self.args.epochs_per_vid == 0):
            print(f"Epoch {self.epochs}.")
        else:
            print(f"{self.epochs}", end = "... ")
            
        # I keep these in training-mode.
        self.gen.train()
        for d in self.dis_list:
            d.train()

        # Generate images for training discriminators.
        with torch.no_grad():
            fake_images, _, _ = self.gen()
        
        # Collect real images, too.
        real_images = get_random_batch(batch_size = self.args.batch_size)
        
        # Make labels.
        fake_labels = torch.zeros(self.args.batch_size, 1).to(self.args.device)
        real_labels = torch.empty(self.args.batch_size, 1).uniform_(self.args.min_real, self.args.max_real).to(self.args.device)
        
        # Flip some labels.
        original_fake_labels = fake_labels[:self.args.flips].clone()
        original_real_labels = real_labels[:self.args.flips].clone()
        fake_labels[:self.args.flips] = original_real_labels
        real_labels[:self.args.flips] = original_fake_labels
        
        # Train discriminators.
        for dis, opt in zip(self.dis_list, self.dis_opts):
            opt.zero_grad()
            
            # Process fake images.
            output_fake, mu_fake, std_fake = dis(fake_images.detach())  
            complexity_fake_loss = ((std_fake - self.args.min_dis_std) ** 2).mean()
            correct_fake = ((output_fake < .5) == (fake_labels < .5)).float().mean().item()
            loss_fake = F.binary_cross_entropy(output_fake, fake_labels)
            
            # Process real iamges.
            output_real, mu_real, std_real = dis(real_images)
            complexity_real_loss = ((std_real - self.args.min_dis_std) ** 2).mean()
            correct_real = ((output_real > .5) == (real_labels > .5)).float().mean().item()
            loss_real = F.binary_cross_entropy(output_real, real_labels)
            
            # Process loss-values.
            loss = loss_real + loss_fake 
            complexity_loss = self.args.dis_alpha * (complexity_real_loss + complexity_fake_loss)
            loss += complexity_loss     # Discriminator encouraged to have minimize its entropy.
            loss.backward()
            opt.step()
            
            torch.cuda.empty_cache()
            
            # Save information.
            self.plot_vals_dict["dis_correct_rate_fake"][-1].append(correct_fake)
            self.plot_vals_dict["dis_losses_fake"][-1].append(loss_fake.item())
            self.plot_vals_dict["dis_correct_rate_real"][-1].append(correct_real)
            self.plot_vals_dict["dis_losses_real"][-1].append(loss_real.item())
            self.plot_vals_dict["dis_complexity_loss"][-1].append(complexity_loss.item())
            self.plot_vals_dict["dis_mu"][-1].append((mu_real.mean().item() + mu_fake.mean().item())/2)
            self.plot_vals_dict["dis_std"][-1].append((std_real.mean().item() + std_fake.mean().item())/2)

        # Generate images for training generator. Label them as real, so the generator learns to trick discriminator.
        fake_images, mu, std = self.gen()
        real_labels = torch.ones(self.args.batch_size, 1).to(self.args.device)
        
        # Get entropy value.
        entropy_loss = -0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * std**2)
        entropy_loss = self.args.alpha * entropy_loss.mean()
        
        # Make discriminators judge the generated images.
        self.gen_opt.zero_grad()
        loss_g = torch.tensor(0.0).to(self.args.device)
        curiosity_loss = torch.tensor(0.0).to(self.args.device)
        for dis in self.dis_list:
            output_fake, mu_new_fake, std_new_fake = dis(fake_images)
            loss_g += self.args.extrinsic * F.binary_cross_entropy(output_fake, real_labels) / len(self.dis_list)
            # Could the generator confuse the discriminator?
            curiosity_loss += -self.args.beta * std_new_fake.mean() / len(self.dis_list)

        # Same informaiton.
        self.plot_vals_dict["gen_loss"].append(loss_g.item())
        self.plot_vals_dict["gen_entropy_loss"].append(entropy_loss.item())
        self.plot_vals_dict["gen_curiosity_loss"].append(curiosity_loss.item())
        self.plot_vals_dict["gen_mu"].append(mu.mean().item()) 
        self.plot_vals_dict["gen_std"].append(std.mean().item())
        
        loss_g += entropy_loss       # Generator encouraged to maximize entropy.
        loss_g += curiosity_loss     # Generator encouraged to make the discriminator have complexity.
        loss_g.backward()
        self.gen_opt.step()
        
        torch.cuda.empty_cache()
        
        # Every once in a while, plot generated images and make a video of them. Plot loss values, etc.
        if(self.epochs % self.args.epochs_per_vid == 0):
            self.make_images_with_seeds()
            plot_vals(self.plot_vals_dict, save_path = f'{self.args.arg_name}/epoch_{str(self.epochs).zfill(5)}/losses.png')
            print(duration())
            
            torch.cuda.empty_cache()
        
        # Add lists to this dictionary, for tracking multiple discriminators. 
        self.plot_vals_dict["dis_losses_real"].append([])
        self.plot_vals_dict["dis_losses_fake"].append([])
        self.plot_vals_dict["dis_correct_rate_real"].append([])
        self.plot_vals_dict["dis_correct_rate_fake"].append([])
        self.plot_vals_dict["dis_complexity_loss"].append([])
        self.plot_vals_dict["dis_mu"].append([])
        self.plot_vals_dict["dis_std"].append([])
        
        self.epochs += 1
        
        if self.epochs == self.args.epochs:
            pass
    
    # Make those pictures and animate them.
    def make_images_with_seeds(self):
        fake_images, _, _ = self.gen(self.seeds, use_std = False)
        show_images_from_tensor(fake_images.unsqueeze(0), save_path=f'{self.args.arg_name}/epoch_{str(self.epochs).zfill(5)}')
        
    # Let's do this!
    def training(self):
        for epoch in range(self.args.epochs):
            self.epoch()
            percent_done = str(self.epochs / self.args.epochs)
        # When training is over, make animations across all epochs.
        make_animation(
            save_dir = r"C:\Users\Ted\Desktop\FEP_Blorpomon\generated_images" + "\\" + f"{self.args.arg_name}",
            image_name='1.png', 
            output_name='through_training.gif')
        make_animation(
            save_dir = r"C:\Users\Ted\Desktop\FEP_Blorpomon\generated_images" + "\\" + f"{self.args.arg_name}", 
            image_name='losses.png', 
            output_name='all_losses.gif')
                
        
        
# :D
if(__name__ == "__main__"):
    gan = GAN()
    gan.training()
    
    
    
    
