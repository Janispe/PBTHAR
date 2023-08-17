
from dataloaders import data_set,data_dict
import torch
import yaml
import os
import torch.nn as nn

import torch
import torch.optim as optim

from dataloaders import data_set,data_dict
from models.model_builder import model_builder

from pbtexperiment import _get_data, validation

def build_model(args, device, checkpoint_dict):
    model = model_builder(args)
    model.double()
    model.to(device)
    
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return model

def cal_test_accuracy(args, device, checkpoint_dict):
    model = build_model(args, device, checkpoint_dict)
    
    dataset = data_dict[args.data_name](args)
    dataset.update_train_val_test_keys()
    
    test_dataloader = _get_data(args, dataset, flag = 'test', weighted_sampler = args.weighted_sampler )
    return validation(args, model, test_dataloader, device)