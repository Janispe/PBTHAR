#!/bin/bash
export PYTHONPATH="/usr/bin/python3"

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api class --training-iterations 75 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "00" --synch True --use-vali-keys True \
            --early-stopping True