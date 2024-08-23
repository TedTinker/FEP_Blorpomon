#%%

import os
import pickle, torch, random
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from time import sleep 
from math import floor
from copy import deepcopy

from gan import GAN
from utils import args, duration, estimate_total_duration, print



print("\nname:\n{}".format(args.arg_name))
print("\ntitle:\n{}".format(args.arg_title))



if __name__ == '__main__':    
    seed = args.init_seed #  + i
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
            
    gan = GAN(args = args)
    gan.training()
    
    print("\nDuration: {}. Done!".format(duration()))
    # %%
