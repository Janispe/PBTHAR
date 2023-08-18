#!/bin/bash
export PYTHONPATH="/usr/bin/python3"

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api class --training-iterations 50 --reuse-actor True \
            --pertubation-interval 5 --experiment-name "00" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api class --training-iterations 100 --reuse-actor False \
            --pertubation-interval 5 --experiment-name "01" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api class --training-iterations 100 --reuse-actor True \
            --pertubation-interval 5 --experiment-name "02" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "03" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 4 --experiment-name "04" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 3 --experiment-name "05" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 8 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "06" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 8 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "99" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "07" --synch False --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "08" --synch True --use-vali-keys False

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "09" --synch True --use-vali-keys False

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 4 --experiment-name "10" --synch True --use-vali-keys False

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 150 \
            --pertubation-interval 4 --experiment-name "11" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 4 --experiment-name "12" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 4 --experiment-name "13" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 100 \
            --pertubation-interval 5 --experiment-name "14" --synch True --use-vali-keys True

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 150 \
            --pertubation-interval 5 --experiment-name "15" --synch True --use-vali-keys False