#!/bin/bash
export PYTHONPATH="/usr/bin/python3"


python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-7" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4


python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 30 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 40 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 50 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "-1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 40 --reuse-actor True \
            --pertubation-interval 7 --experiment-name "00" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 40 --reuse-actor True \
            --pertubation-interval 7 --experiment-name "01" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 40 --reuse-actor True \
            --pertubation-interval 7 --experiment-name "02" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "03" --synch True --use-vali-keys True \
            --difference True --filtering True --custom-start-values 1 --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "04" --synch True --use-vali-keys True \
            --difference True --filtering True --custom-start-values 1 --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor False \
            --pertubation-interval 2 --experiment-name "05" --synch True --use-vali-keys True \
            --difference True --filtering True --custom-start-values 1 --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor False \
            --pertubation-interval 2 --experiment-name "06" --synch True --use-vali-keys True \
            --difference True --filtering True --custom-start-values 1 --seed 4

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "07" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "08" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "09" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name dg --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 35 --reuse-actor True \
            --pertubation-interval 2 --experiment-name "10" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4


python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 30 --reuse-actor True \
            --pertubation-interval 3 --experiment-name "12" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 40 --reuse-actor True \
            --pertubation-interval 3 --experiment-name "13" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt" --data-name oppo --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 50 --reuse-actor True \
            --pertubation-interval 3 --experiment-name "11" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

