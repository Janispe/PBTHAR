#!/bin/bash
export PYTHONPATH="/usr/bin/python3"


python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6

