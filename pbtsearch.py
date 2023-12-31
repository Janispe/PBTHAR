from functionapi import train_net

from dataloaders import data_dict

from ray import tune, air
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune.stopper import ExperimentPlateauStopper, MaximumIterationStopper, CombinedStopper

from PBTTrainable import RayModel

import numpy as np

import random

def train(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
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
            "random_aug_first" : list(np.arange(0,1.1,0.1))
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
            "random_augmentation_prob": [0,1],
            "mixup_probability": [0,1],
            "random_aug_first":[0,1]
        },
        synch=args.synch
        )
    else:
        raise ValueError("no " + args.scheduler)

    dataset = data_dict[args.data_name](args)
    dataset.update_train_val_test_keys()
    
    if args.trainable_api=='function':
        trainable = train_net
    elif args.trainable_api=='class':
        trainable = RayModel
    else:
        raise AttributeError()
    
    trainable = tune.with_parameters(trainable, dataset=dataset, args=args)
    trainable_with_resources = tune.with_resources(trainable, {"gpu": args.gpu_per_trial, "cpu": args.cpu_per_trial})

    if args.early_stopping:
        stop=CombinedStopper(MaximumIterationStopper(args.training_iterations),
                              ExperimentPlateauStopper('total_loss', mode="min", patience=args.early_stop_patience,
                                                       top=2))
    else:
        stop={"training_iteration": args.training_iterations}

    if args.custom_start_values is not None:
        param_space = {
                "random_augmentation_prob": args.custom_start_values,
                "mixup_probability": args.custom_start_values,
                "random_aug_first": args.custom_start_values,
                "checkpoint_interval": args.custom_start_values,
                "seed": tune.randint(0, 10000),
        }
    else:
        param_space = {
                "random_augmentation_prob": tune.choice(list(np.arange(0,1.1,0.1))),
                "mixup_probability": tune.choice(list(np.arange(0,1.1,0.1))),
                "random_aug_first": tune.choice(list(np.arange(0,1.1,0.1))),
                "checkpoint_interval": args.perturbation_interval,
                "seed": tune.randint(0, 10000),
            }

    if args.restore:
        tuner = tune.Tuner.restore(args.storage_path)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            run_config=air.RunConfig(
                log_to_file=True,
                name=args.experiment_name,
                stop=stop,
                verbose=1,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute="mean_accuracy",
                    num_to_keep=4,
                ),
                storage_path=args.storage_path,
            ),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=args.trials,
                reuse_actors=args.reuse_actor,
                chdir_to_trial_dir=False,
            ),
            param_space=param_space,
        )
    

    #if ray.is_initialized():
    #    ray.shutdown()
    #ray.init(runtime_env={"working_dir":"","excludes":["datasets"]})

    return tuner.fit()