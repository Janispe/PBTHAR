#!/bin/bash
export PYTHONPATH="/usr/bin/python3"

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6





python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/hapt_final/attend" --data-name hapt --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5 --model-type attend

python main.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/pamap2_final/attend" --data-name pamap2 --trials 16 \
            --gpu-per-trial 0.2 --trainable-api function --training-iterations 60 --reuse-actor True \
            --pertubation-interval 4 --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6 --model-type attend

