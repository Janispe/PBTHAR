
from dataloaders import data_dict

import matplotlib.pyplot as plt

import pandas as pd
import collections
import json
import ast

from dataloaders import data_dict
from models.model_builder import model_builder

from pbtexperiment import _get_data, validation
from utils import MixUpLoss

import torch.nn as nn

from typing import  Dict, List,  Tuple

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

PbtUpdate = collections.namedtuple('PbtUpdate', [
    'target_trial_name', 'clone_trial_name', 'target_trial_epochs',
    'clone_trial_epochs', 'old_config', 'new_config'
])

def _load_policy(policy_file: str) -> Tuple[Dict, List[Tuple[int, Dict]]]:
    raw_policy = []
    with open(policy_file, "rt") as fp:
        for row in fp.readlines():
            try:
                parsed_row = json.loads(row)
            except json.JSONDecodeError:
                raise ValueError(
                    "Could not read PBT policy file: {}.".format(policy_file)
                ) from None
            raw_policy.append(tuple(parsed_row))

    # Loop through policy from end to start to obtain changepoints
    policy = []
    last_new_tag = None
    last_old_conf = None
    for old_tag, new_tag, old_step, new_step, old_conf, new_conf in reversed(
        raw_policy
    ):
        if last_new_tag and old_tag != last_new_tag:
            # Tag chain ended. This means that previous changes were
            # overwritten by the last change and should be ignored.
            break
        last_new_tag = new_tag
        last_old_conf = old_conf

        policy.append((new_step, new_conf))

    return last_old_conf, list(reversed(policy))