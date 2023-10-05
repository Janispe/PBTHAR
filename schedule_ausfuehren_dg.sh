python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/1/pbt_policy_85702_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "1" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 1 --early-stop-patience 15

python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/2/pbt_policy_42523_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "2" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 2 --early-stop-patience 15

python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/3/pbt_policy_34041_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "3" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 3 --early-stop-patience 15

python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/4/pbt_policy_57199_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "4" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 4 --early-stop-patience 15

python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/5/pbt_policy_a637c_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "5" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 5 --early-stop-patience 15

python replay.py --storage-path "/home/janis/PopulationBasedTraining/RayTuneResults/dg/schedule" --data-name dg --trials 1 \
            --policy-file "/home/janis/PopulationBasedTraining/RayTuneResults/dg/pbt/6/pbt_policy_026b4_00000.txt"  \
            --gpu-per-trial 1 --trainable-api function --training-iterations 100 --reuse-actor True \
             --experiment-name "6" --synch True --use-vali-keys True \
            --difference True --filtering True --seed 6 --early-stop-patience 15

