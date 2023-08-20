
from setup import create_parser

from dataloaders import  data_dict
import numpy as np
import yaml

import torch
import os

from pbtsearch import train

from analysis import cal_test_accuracy


args = create_parser().parse_args()

config_file = open('configs/data.yaml', mode='r')
data_config = yaml.load(config_file, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path       =  os.path.join(args.root_path,config["filename"])
args.sampling_freq   =  config["sampling_freq"]
args.num_classes     =  config["num_classes"]
window_seconds       =  config["window_seconds"]
args.windowsize      =  int(window_seconds * args.sampling_freq) 
args.input_length    =  args.windowsize
# input information
args.c_in            =  config["num_channels"]

if args.difference:
    args.c_in  = args.c_in * 2
if  args.filtering :
    for col in config["sensors"]:
        if "acc" in col:
            args.c_in = args.c_in+1
            
if args.wavelet_filtering :
    if args.windowsize%2==1:
        N_ds = int(torch.log2(torch.tensor(args.windowsize-1)).floor()) - 2
    else:
        N_ds = int(torch.log2(torch.tensor(args.windowsize)).floor()) - 2

    args.f_in            =  args.number_wavelet_filtering*N_ds+1
else:
    args.f_in            =  1


args.random_augmentation_config = {"jitter":True,
                                "moving_average":True,
                                "magnitude_scaling":True,
                                "magnitude_warp":True,
                                "magnitude_shift":True,
                                "time_warp":True,
                                "window_warp":True,
                                "window_slice":True,
                                "random_sampling":True,
                                "slope_adding":True
                                }
random_augmentation_nr = 0
for key in args.random_augmentation_config.keys():
    if args.random_augmentation_config[key]:
        random_augmentation_nr = random_augmentation_nr+1
args.random_augmentation_nr = random_augmentation_nr

args.exp_mode = "Given"

results_grid = train(args)

best_result = results_grid.get_best_result(mode="max", metric="mean_accuracy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = best_result.checkpoint
if args.trainable_api=='function':
    dict = checkpoint.to_dict()
elif args.trainable_api=='class':
    dict = torch.load(checkpoint._local_path+'/model.pht')
else:
    raise AttributeError()

total_loss,  acc, f_w,  f_macro, f_micro = cal_test_accuracy(args, device, dict)
text = 'Best Trial Test Performance: Total_Loss {} Acc {}  F_w {} F_Macro {} F_Micro {}'.format(total_loss,  acc, f_w,  f_macro, f_micro)
print(text)

path = args.storage_path + '/' + args.experiment_name + '/best_trial_final_test_results.txt'
f = open(path, "w")
f.write(text)
f.close()


for r in results_grid:
    checkpoint = r.checkpoint
    if args.trainable_api=='function':
        dict = checkpoint.to_dict()
    elif args.trainable_api=='class':
        dict = torch.load(checkpoint._local_path+'/model.pht')
    else:
        raise AttributeError()

    total_loss,  acc, f_w,  f_macro, f_micro = cal_test_accuracy(args, device, dict)
    text = 'Trial Test Performance: Total_Loss {} Acc {}  F_w {} F_Macro {} F_Micro {}'.format(total_loss,  acc, f_w,  f_macro, f_micro)
    
    path = r._local_path + '/trial_final_test_results.txt'
    f = open(path, "w")
    f.write(text)
    f.close()
