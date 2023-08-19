
from dataloaders import data_dict

import matplotlib.pyplot as plt

import pandas as pd

from dataloaders import data_dict
from models.model_builder import model_builder

from pbtexperiment import _get_data, validation
from utils import MixUpLoss

import torch.nn as nn

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
    
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    criterion = MixUpLoss(criterion)
    
    return validation(args, model, test_dataloader, criterion, device)

def plot_hp_history(results, parameter):
    fig, ax = plt.subplots()
    ax.set_title(parameter + "  over training iterations")
    ax.set_xlabel("training_iteration")
    ax.set_ylabel(parameter)
    for i in range(len(results)):
        df = results[i].metrics_dataframe
        ax.plot(df[parameter])
    ax.legend()
    
def get_average_frame(results_grid, parameter):
    dataframes = []
    for rg in results_grid:
        dataframes.append(rg.metrics_dataframe[["training_iteration", parameter]])

    df = pd.concat(dataframes)
    avg_df = df.groupby("training_iteration")
    return avg_df.mean()