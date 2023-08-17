import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import time
from dataloaders import data_dict,data_set
from sklearn.metrics import confusion_matrix
import yaml
import pandas as pd

from experiment import MixUpLoss

from models.model_builder import model_builder

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataloaders.augmentation import RandomAugment, mixup_data
import random
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ray.tune import Stopper
from ray.air import session, Checkpoint

import ray
from ray import tune, air
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2


def _get_data(args, data, flag="train", weighted_sampler = False):
    if flag == 'train':
        shuffle_flag = True 
    else:
        shuffle_flag = False

    data  = data_set(args,data,flag)


    if flag == "train":
        if args.mixup_probability < 1 or args.random_augmentation_prob<1:
            random_aug =RandomAugment(args.random_augmentation_nr, args.random_augmentation_config, args.max_aug )
            def collate_fn(batch):                
                if (args.random_aug_first):
                    batch_x1 = []
                    batch_x2 = []
                    batch_y  = []
                    for x,y,z in batch:

                        if np.random.uniform(0,1,1)[0]>=args.random_augmentation_prob:
                            batch_x1.append(random_aug(x))
                        else:
                            batch_x1.append(x)
                        batch_x2.append(y)
                        batch_y.append(z)

                    batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x2 = torch.tensor(batch_x2)
                    batch_y = torch.tensor(batch_y)
                    
                    if np.random.uniform(0,1,1)[0] >= args.mixup_probability:

                        batch_x1 , batch_y = mixup_data(batch_x1 , batch_y,   args.mixup_alpha,  argmax = args.mixup_argmax)
                        #print("Mixup",batch_x1.shape,batch_y.shape)
                    #else:
                        #print(batch_x1.shape,batch_y.shape)
                    batch_x1 = torch.unsqueeze(batch_x1,1)
                    return batch_x1,batch_x2,batch_y
                else:
                    batch_x1 = []
                    batch_x2 = []
                    batch_y  = []
                    for x,y,z in batch:
                        batch_x1.append(x)
                        batch_x2.append(y)
                        batch_y.append(z)
                    
                    batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x2 = torch.tensor(batch_x2)
                    batch_y = torch.tensor(batch_y)

                    if np.random.uniform(0,1,1)[0] >= args.mixup_probability:
                        batch_x1 , batch_y = mixup_data(batch_x1 , batch_y,   args.mixup_alpha,  argmax = args.mixup_argmax)

                    batch_x1 = batch_x1.detach().cpu().numpy()

                    batch_x1_list = []
                    for x in batch_x1:
                        x = x[np.newaxis]
                        if np.random.uniform(0,1,1)[0]>=args.random_augmentation_prob:
                            batch_x1_list.append(random_aug(x))
                        else:
                            batch_x1_list.append(x)
                    #batch_x1 = torch.tensor(np.concatenate(batch_x1, axis=0))
                    batch_x1 = torch.tensor(batch_x1)

                    batch_x1 = torch.unsqueeze(batch_x1,1)

                    return batch_x1,batch_x2,batch_y


                    
        else:
            collate_fn = None
    else:
        collate_fn = None

    if weighted_sampler and flag == 'train':

        sampler = WeightedRandomSampler(
            data.act_weights, len(data.act_weights)
        )

        data_loader = DataLoader(data, 
                                batch_size   =  args.batch_size,
                                #shuffle      =  shuffle_flag,
                                num_workers  =  0,
                                sampler=sampler,
                                drop_last    =  False,
                                collate_fn = collate_fn)
    else:
        data_loader = DataLoader(data, 
                                batch_size   =  args.batch_size,
                                shuffle      =  shuffle_flag,
                                num_workers  =  0,
                                drop_last    =  False,
                                collate_fn   = collate_fn)
    return data_loader

def validation(args, model, data_loader, criterion, device, index_of_cv=None, selected_index = None):
    model.eval()
    total_loss = []
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x1,batch_x2,batch_y) in enumerate(data_loader):

            if "cross" in args.model_type:
                batch_x1 = batch_x1.double().to(device)
                batch_x2 = batch_x2.double().to(device)
                batch_y = batch_y.long().to(device)
                # model prediction
                if args.output_attention:
                    outputs = model(batch_x1,batch_x2)[0]
                else:
                    outputs = model(batch_x1,batch_x2)
            else:
                if selected_index is None:
                    batch_x1 = batch_x1.double().to(device)
                else:
                    batch_x1 = batch_x1[:, selected_index.tolist(),:,:].double().to(device)
                batch_y = batch_y.long().to(device)

                # model prediction
                if args.output_attention:
                    outputs = model(batch_x1)[0]
                else:
                    outputs = model(batch_x1)


            pred = outputs.detach()#.cpu()
            true = batch_y.detach()#.cpu()

            loss = criterion(pred, true) 
            total_loss.append(loss.cpu())
            
            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy()))   
            
    total_loss = np.average(total_loss)
    acc = accuracy_score(preds,trues)
    #f_1 = f1_score(trues, preds)
    f_w = f1_score(trues, preds, average='weighted')
    f_macro = f1_score(trues, preds, average='macro')
    f_micro = f1_score(trues, preds, average='micro')
    if index_of_cv:
        cf_matrix = confusion_matrix(trues, preds)
        #with open("{}.npy".format(index_of_cv), 'wb') as f:
        #    np.save(f, cf_matrix)
        plt.figure()
        sns.heatmap(cf_matrix, annot=True)
        #plt.savefig("{}.png".format(index_of_cv))
    model.train()

    return total_loss,  acc, f_w,  f_macro, f_micro#, f_1


def train_net(config, dataset, args):
    print(os.path.abspath(os.curdir))
    #args = config["args"]
    #args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True 
    #np.random.seed(args.seed)

    args.random_augmentation_prob = config["random_augmentation_prob"]
    args.mixup_probability = config["mixup_probability"]
    args.random_aug_first = config["random_aug_first"]

    step = 1
    #dataset = data_dict[args.data_name](args)
    #dataset.update_train_val_test_keys()
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
            })


        acc2 = acc
        session.report(
            {"mean_accuracy": acc, "total_loss": total_loss, "random_augmentation_prob": config["random_augmentation_prob"]
            ,"mixup_probability":config["mixup_probability"], "random_aug_first":config["random_aug_first"]},
            checkpoint=checkpoint
        )
        step += 1


def trainstep(model, optimizer, train_loader, criterion, device):
    model.train()
    for i, (batch_x1,batch_x2,batch_y) in enumerate(train_loader):
        batch_x1 = batch_x1.double().to(device)
        batch_y = batch_y.long().to(device)

        outputs = model(batch_x1)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

def train(args):
    if args.scheduler == "pbt":
        #Population based training
        scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=args.perturbation_interval,
        burn_in_period=args.burn_in_period,
        metric="mean_accuracy",
        quantile_fraction=args.quantile_fraction,
        mode="max",
        resample_probability=args.resample_probability,
        hyperparam_mutations={
            "random_augmentation_prob": list(np.arange(0,1.1,0.1)),
            "mixup_probability": list(np.arange(0,1.1,0.1)),
            #"random_aug_first" : [True, False]
        },
        synch=args.synch,
        )
    elif args.scheduler == "pbt2":
        #Population based bandits
        scheduler = PB2(
        time_attr='training_iteration',
        metric='mean_accuracy',
        mode='max',
        quantile_fraction=args.quantile_fraction,
        perturbation_interval=args.perturbation_interval,
        hyperparam_bounds={
            "random_augmentation_prob": [0.1,0.9],
            #"mixup_probability": [0.1,0.9],
            #"random_aug_first":[True, False]
        },
        synch=args.synch
        )
    else:
        raise ValueError("no " + args.scheduler)


    #experiment_name = args.data_name + "_" + str(args.perturbation_interval) +"_1"
    storage_path = args.storage_path

    dataset = data_dict[args.data_name](args)
    dataset.update_train_val_test_keys()
    
    trainable = tune.with_parameters(train_net, dataset=dataset, args=args)
    trainable_with_resources = tune.with_resources(trainable, {"gpu": args.gpu_per_trial, "cpu": args.cpu_per_trial})


    if args.restore:
        tuner = tune.Tuner.restore(storage_path)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            run_config=air.RunConfig(
                #name=experiment_name,
                # Stop when we've reached a threshold accuracy, or a maximum
                # training_iteration, whichever comes first
                stop={"training_iteration": args.training_iterations},
                verbose=1,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=4,
                ),
                storage_path=storage_path,
            ),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=args.trials,
                chdir_to_trial_dir=False,
            ),
            param_space={
                #"random_augmentation_prob": 0.5,
                #"mixup_probability": 0.5,
                "random_augmentation_prob": tune.choice(list(np.arange(0,1.1,0.1))),
                "mixup_probability": tune.choice(list(np.arange(0,1.1,0.1))),
                "random_aug_first": tune.choice([True, False]),
                #"random_aug_first": True,
                "checkpoint_interval": args.perturbation_interval,
    #           "args": args,
            },
        )
    

    if ray.is_initialized():
        ray.shutdown()
    #ray.init(runtime_env={"working_dir":""})
    ray.init(runtime_env={"working_dir":"","excludes":["datasets"]})

    return tuner.fit()


class EarlyStopper(Stopper):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, f_macro = None, f_weighted = None, log=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model, path ,f_macro, f_weighted)

        elif score < self.best_score + self.delta:
            self.counter += 1

            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            #print("new best score!!!!")
            #print("log shi ", log)
            #if log is not None:
                #log.write("new best score!!!! Saving model ... \n")
                #log.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n")
            self.best_score = score
            #self.save_checkpoint(val_loss, model,path,f_macro, f_weighted)
            self.counter = 0
        return self.early_stop
    
class adjust_learning_rate_class:
    def __init__(self, args, verbose):
        self.patience = args.learning_rate_patience
        self.factor   = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.args = args
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_score = None
    def __call__(self, optimizer, val_loss):
        # val_loss 是正值，越小越好
        # 但是这里加了负值，score愈大越好
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score :
            self.counter += 1
            if self.verbose:
                print(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                print("new best score!!!!")
            self.best_score = score
            self.counter = 0
            
        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    print('Updating learning rate to {}'.format(self.learning_rate))
            self.counter = 0
    def state_dict(self):
        return {'patience':self.patience, 'factor':self.factor, 'learning_rate':self.learning_rate,
                'verbose':self.verbose, 'counter':self.counter, 'best_score':self.best_score}
    def load_state_dict(self, dict):
        self.patience = dict['patience']
        self.factor = dict['factor']
        self.learning_rate = dict['learning_rate']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        
