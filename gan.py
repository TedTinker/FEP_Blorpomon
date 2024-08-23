#%% 
import os

import torch 
from torch.optim import Adam
import torch.nn.functional as F

from utils import default_args, get_random_batch, create_interpolated_tensor, show_images_from_tensor, plot_vals, print
from generator import Generator
from discriminator import Discriminator



class GAN:
    def __init__(self, args = default_args):
        self.args = args
        
        folder_name = "generated_images/" + str(self.args.arg_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), args.gen_lr)
        
        self.dis_list = [Discriminator() for i in range(self.args.dises)]
        self.dis_opts = [Adam(dis.parameters(), args.dis_lr) for dis in self.dis_list]
        
        self.seeds = create_interpolated_tensor(self.args).to(self.args.device)
        
        self.plot_vals_dict = {
            "gen_loss" : [],
            "dis_losses_real" : [[]],
            "dis_losses_fake" : [[]],
            "dis_correct_rate_real" : [[]],
            "dis_correct_rate_fake" : [[]]}
        
        self.epochs = 0
        
    def epoch(self):
        if(self.epochs % self.args.epochs_per_vid == 0):
            print(f"Epoch {self.epochs}")
        self.gen.train()
        for d in self.dis_list:
            d.train()

        # Make data
        fake_images, _ = self.gen()
        fake_labels = torch.zeros(self.args.batch_size, 1).to(self.args.device)
        
        real_images = get_random_batch(batch_size = self.args.batch_size)
        real_labels = torch.ones(self.args.batch_size, 1).to(self.args.device)
        
        # Train discriminators
        for dis, opt in zip(self.dis_list, self.dis_opts):
            opt.zero_grad()
            
            output_fake, log_prob_fake = dis(fake_images.detach())  
            correct_fake = ((output_fake < .5) == (fake_labels < .5)).float().mean().item()
            self.plot_vals_dict["dis_correct_rate_fake"][-1].append(correct_fake)
            loss_fake = F.binary_cross_entropy(output_fake, fake_labels)
            self.plot_vals_dict["dis_losses_fake"][-1].append(loss_fake.item())
            
            output_real, log_prob_real = dis(real_images)
            correct_real = ((output_real > .5) == (real_labels > .5)).float().mean().item()
            self.plot_vals_dict["dis_correct_rate_real"][-1].append(correct_real)
            loss_real = F.binary_cross_entropy(output_real, real_labels)
            self.plot_vals_dict["dis_losses_real"][-1].append(loss_real.item())
            
            loss = loss_real + loss_fake + self.args.dis_alpha * (log_prob_fake.mean() + log_prob_real.mean())
            loss.backward()
            opt.step()
            
        # Make data
        fake_images, log_prob_gen = self.gen()
        real_labels = torch.ones(self.args.batch_size, 1).to(self.args.device)
        
        # Train generator
        self.gen_opt.zero_grad()
        loss_g = torch.tensor(0.0).to(self.args.device)
        for dis in self.dis_list:
            output_fake, log_prob_new_fake = dis(fake_images)
            loss_g += self.args.extrinsic * F.binary_cross_entropy(output_fake, real_labels) / len(self.dis_list)
            loss_g -= self.args.beta * log_prob_new_fake.mean() / len(self.dis_list)
        loss_g += self.args.alpha * log_prob_gen.mean()
        loss_g.backward()
        self.gen_opt.step()
        self.plot_vals_dict["gen_loss"].append(loss_g.item())
        
        if(self.epochs % self.args.epochs_per_vid == 0):
            self.make_images_with_seeds()
        if( self.epochs % self.args.epochs_per_vid == 0):
            plot_vals(self.plot_vals_dict, save_path = f'{self.args.arg_name}/losses.png')
        
        self.plot_vals_dict["dis_losses_real"].append([])
        self.plot_vals_dict["dis_losses_fake"].append([])
        self.plot_vals_dict["dis_correct_rate_real"].append([])
        self.plot_vals_dict["dis_correct_rate_fake"].append([])
        
        self.epochs += 1
        
    
    
    def make_images_with_seeds(self):
        fake_images, _ = self.gen(self.seeds, use_std = False)
        show_images_from_tensor(fake_images.unsqueeze(0), save_path=f'{self.args.arg_name}/animation_{self.epochs}.gif')
        
    def training(self):
        for epoch in range(default_args.epochs):
            self.epoch()
            percent_done = str(self.epochs / self.args.epochs)
                
        
        
if(__name__ == "__main__"):
    gan = GAN()
    gan.training()
# %%
