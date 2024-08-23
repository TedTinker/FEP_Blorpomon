import os
os.chdir("C://Users//tedjt//Desktop//Thinkster//126 fep gan/fep gan")

import torch 
from torch.optim import Adam
import torch.nn.functional as F

from utils import default_args, get_random_batch, create_interpolated_tensor, show_images_from_tensor, plot_vals
from generator import Generator
from discriminator import Discriminator



class GAN:
    def __init__(self, args = default_args):
        self.args = default_args
        
        self.gen = Generator()
        self.gen_opt = Adam(self.gen.parameters(), args.gen_lr)
        
        self.dis_list = [Discriminator() for i in range(self.args.dises)]
        self.dis_opts = [Adam(dis.parameters(), args.dis_lr) for dis in self.dis_list]
        
        self.seeds = create_interpolated_tensor(self.args)
        
        self.plot_vals_dict = {
            "gen_loss" : [],
            "dis_losses_real" : [[]],
            "dis_losses_fake" : [[]],
            "dis_correct_rate_real" : [[]],
            "dis_correct_rate_fake" : [[]]}
        
        self.epochs = 0
        
    def epoch(self):
        print(self.epochs, end = ", ")
        self.gen.train()
        for d in self.dis_list:
            d.train()

        # Make data
        fake_images, _ = self.gen()
        fake_labels = torch.zeros(self.args.batch_size, 1)
        
        real_images = get_random_batch(batch_size = self.args.batch_size)
        real_labels = torch.ones(self.args.batch_size, 1)
        
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
            
            loss = loss_real + loss_fake + self.args.dis_alpha * (loss_fake.mean() + loss_real.mean())
            loss.backward()
            opt.step()
            
        # Make data
        fake_images, log_prob_gen = self.gen()
        real_labels = torch.ones(self.args.batch_size, 1)
        
        # Train generator
        self.gen_opt.zero_grad()
        loss_g = torch.tensor(0.0)
        for dis in self.dis_list:
            output_fake, log_prob_new_fake = dis(fake_images)
            loss_g += self.args.extrinsic * F.binary_cross_entropy(output_fake, real_labels) 
            loss_g -= self.args.beta * log_prob_new_fake.mean()
        loss_g += self.args.alpha * log_prob_gen.mean()
        loss_g.backward()
        self.gen_opt.step()
        self.plot_vals_dict["gen_loss"].append(loss_g.item())
        
        if(self.epochs % self.args.epochs_per_vid == 0):
            self.make_images_with_seeds()
        if( self.epochs % self.args.epochs_per_vid == 0):
            plot_vals(self.plot_vals_dict)
        
        self.plot_vals_dict["dis_losses_real"].append([])
        self.plot_vals_dict["dis_losses_fake"].append([])
        self.plot_vals_dict["dis_correct_rate_real"].append([])
        self.plot_vals_dict["dis_correct_rate_fake"].append([])
        
        self.epochs += 1
        
    
    def make_images_with_seeds(self):
        fake_images, _ = self.gen(self.seeds, use_std = False)
        show_images_from_tensor(fake_images.unsqueeze(0), save_path=f'animation_{self.epochs}.gif')
        
                
        
        
if(__name__ == "__main__"):
    gan = GAN()
    for epoch in range(default_args.epochs):
        gan.epoch()