from ray.tune import Trainable
import torch
from torch import optim
import torch.nn as nn
import os

from pbtexperiment import _get_data, trainstep, validation
from utils import adjust_learning_rate_class
from models.model_builder import model_builder

from experiment import MixUpLoss


class RayModel(Trainable):
    
    def setup(self, config: dict, dataset, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        
        self.args.random_augmentation_prob = config["random_augmentation_prob"]
        self.args.mixup_probability = config["mixup_probability"]
        self.args.random_aug_first = config["random_aug_first"]
        
        self.train_loader = _get_data(args, dataset, flag = 'train', weighted_sampler = args.weighted_sampler )
        self.vali_loader = _get_data(args, dataset, flag = 'vali', weighted_sampler = args.weighted_sampler )
        
        self.model = model_builder(args)
        self.model.double()
        self.model.to(self.device)
        
        optimizer_dict = {"Adam":optim.Adam}

        self.optimizer = optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.criterion = MixUpLoss(self.criterion)
        
        self.learning_rate_adapter = adjust_learning_rate_class(args,True)
        
    def step(self):
        trainstep(self.model, self.optimizer, self.train_loader, self.criterion, self.device)
        total_loss,  acc, f_w,  f_macro, f_micro = validation(self.args, self.model, self.vali_loader, self.criterion, self.device)
        self.learning_rate_adapter(self.optimizer, total_loss)
        return {"mean_accuracy":acc, "total_loss": total_loss, "random_augmentation_prob": self.args.random_augmentation_prob,
                "mixup_probability": self.args.mixup_probability, "random_aug_first": self.args.random_aug_first}
        
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pht")
        torch.save({"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                    'learning_reate_adapter_state_diict': self.learning_rate_adapter.state_dict()}, 
                   checkpoint_path)
        
    def load_checkpoint(self, checkpoint):
        checkpoint_path = os.path.join(checkpoint, "model.pht")
        dict = torch.load(checkpoint_path)
        self.model.load_state_dict(dict["model_state_dict"])
        self.optimizer.load_state_dict(dict["optimizer_state_dict"])
        self.learning_rate_adapter.load_state_dict(dict['learning_reate_adapter_state_diict'])
        
    def reset_config(self, new_config):
        self.args.random_augmentation_prob = new_config["random_augmentation_prob"]
        self.args.mixup_probability = new_config["mixup_probability"]
        self.args.random_aug_first = new_config["random_aug_first"]
        return True