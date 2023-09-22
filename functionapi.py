

import torch
from pbtexperiment import _get_data, trainstep, validation
from models.model_builder import model_builder
from torch import optim
import torch.nn as nn

from utils import MixUpLoss, adjust_learning_rate_class
from ray.air import session, Checkpoint

import numpy as np
import random


def train_net(config, dataset, args):
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True 
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.random_augmentation_prob = config["random_augmentation_prob"]
    args.mixup_probability = config["mixup_probability"]
    args.random_aug_first = config["random_aug_first"]

    step = 1
    train_loader = _get_data(args, dataset, flag = 'train', weighted_sampler = args.weighted_sampler )
    #test_loader = _get_data(args, dataset, flag = 'test', weighted_sampler = args.weighted_sampler )
    vali_loader = _get_data(args, dataset, flag = 'vali', weighted_sampler = args.weighted_sampler )

    model = model_builder(args)
    model.double()
    model.to(device)

    optimizer_dict = {"Adam":optim.Adam}

    optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    criterion = MixUpLoss(criterion)

    learning_rate_adapter = adjust_learning_rate_class(args,True)

    # if checkpoint available resume from it
    if session.get_checkpoint():
        checkpoint_dict = session.get_checkpoint().to_dict()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        learning_rate_adapter.load_state_dict(checkpoint_dict['learning_reate_adapter_state_diict'])

        last_step = checkpoint_dict["step"]
        step = last_step + 1

        #seed loading

        torch.set_rng_state(checkpoint_dict["torch_seed_state"])
        torch.cuda.set_rng_state(checkpoint_dict["torch_cuda_seed_state"])
        torch.cuda.set_rng_state_all(checkpoint_dict["torch_cuda_all_seed_state"])
        np.random.set_state(checkpoint_dict["numpy_seed_state"])
        random.setstate(checkpoint_dict["random_seed_state"])

    while True:
        trainstep(model, optimizer, train_loader, criterion, device)
        total_loss,  acc, f_w,  f_macro, f_micro = validation(args, model, vali_loader, criterion, device)
        learning_rate_adapter(optimizer, total_loss)

        checkpoint = None
        if step % config["checkpoint_interval"] == 0:
            # create new checkpoint
            checkpoint = Checkpoint.from_dict({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'learning_reate_adapter_state_diict': learning_rate_adapter.state_dict(),
                "torch_seed_state": torch.get_rng_state(),
                "torch_cuda_seed_state" : torch.cuda.get_rng_state(),
                "torch_cuda_all_seed_state": torch.cuda.get_rng_state_all(),
                "numpy_seed_state": np.random.get_state(),
                "random_seed_state": random.getstate(),            
            })

        session.report(
            {"mean_accuracy": acc, "total_loss": total_loss, "random_augmentation_prob": config["random_augmentation_prob"]
            ,"mixup_probability":config["mixup_probability"], "random_aug_first":config["random_aug_first"]},
            checkpoint=checkpoint
        )
        step += 1