import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training-iterations', dest='training_iterations', type=int, default=100, help='Number of epochs')
    parser.add_argument('--restore', dest='restore', type=bool, default=False, help="Try to Restore from folder")
    parser.add_argument('--pertubation-interval',dest='perturbation_interval', type=int, default=5, help="Pertubation Interval for Population Based Training")
    parser.add_argument('--trails', dest='trials', type=int, default=1, help="Number of Trials for Population Based Training")
    parser.add_argument('--scheduler', dest='scheduler', default='pbt', choices=('pbt', 'pbt2'))
    parser.add_argument('--storage-path', dest='storage_path', default='/tmp/datasets/', help='Path for Experiment results')
    parser.add_argument('--cpu-per-trial', dest='cpu_per_trial', default=1)
    parser.add_argument('--gpu-per-trial', dest='gpu_per_trial', default=0)
    parser.add_argument('--synch', dest='synch', default=True, help='Synch for Population Based Training, If True Trials synched')
    parser.add_argument('--quantile-fraction', dest='quantile_fraction', default=0.25, help='Bottom Trials to be exploitet')
    
    
    #only pbt
    parser.add_argument('--resample-probability', dest='resample_probability', default=0.2, help='Propability of resampling hyperparameters instead mutating them')
    parser.add_argument('--burn-in-period', dest='burn_in_period', default=0, help='Iterations without elpoition and exploration, used to trai models first')
    
    
    
    parser.add_argument('--freq-save-path', dest='freq_save_path', default="ISWC2022LearnableFilter/Freq_data")
    parser.add_argument('--window-save-path', dest='window_save_path', default="ISWC2022LearnableFilter/Sliding_window")
    parser.add_argument('--root-path', dest='root_path', default="datasets")
    
    parser.add_argument('--drop-transition', dest='drop_transition', default=False)
    parser.add_argument('--datanorm-type', dest='datanorm_type', default='standardization')
    
    parser.add_argument('--batch-size', dest='batch_size', default=256)
    #shuffle ? 
    #drop last ?
    #train_vali_quote ? 
    
    parser.add_argument('--learning-rate', dest='learning_rate', default=0.001)
    parser.add_argument('--learning-rate-patience', dest='learning_rate_patience', default=5)
    parser.add_argument('--learning-rate-factor', dest='learning_rate_factor', default=0.1)
    
    #early_stop_patience ?
    
    parser.add_argument('--optimizer', dest='optimizer', default='Adam')
    parser.add_argument('--criterion', dest='criterion', default='CrossEntropy')
        
    parser.add_argument('--data-name', dest='data_name', default='hapt')
    parser.add_argument('--use-vali-keys', dest='use_vali_keys', default =False)
    
    parser.add_argument('--wavelet-filtering', dest='wavelet_filtering', default=False)
    #wavelet_filtering_regularization ? 
    #wavelet_filetring_finetuning ? 
    
    parser.add_argument('--difference', dest='difference', default=False)
    parser.add_argument('--filtering', dest='filtering', default=False)
    parser.add_argument('--magnitude', dest='magnitude', default=False)
    parser.add_argument('--weighted-sampler', dest='weighted_sampler', default=False)

    parser.add_argument('--pos-select', dest='pos_select', default=None)
    parser.add_argument('--sensor-select', dest='sensor_select', default=None)
    
    parser.add_argument('--representation-type', dest='representation_type', default='time')
    #exp_mode ?
    
    parser.add_argument('--filter-scaling-factor', dest='filter_scaling_factor', default=0.25)
    parser.add_argument('--model-type', dest='model_type', default='deepconvlstm')
    parser.add_argument('--model-config', dest='model_config', default='configs/model.yaml')
    
    #random_aug_first
    parser.add_argument('--mixup-alpha', dest='mixup_alpha', default=0.5)
    parser.add_argument('--mixup-argmax', dest='mixup_argmax', default=True)
    
    parser.add_argument('--max-aug', dest='max_aug', default=3)
    
    
    #needed ?
    parser.add_argument('--load-all', dest='load_all', default=None)
    parser.add_argument('--train-vali-quote', dest='train_vali_quote', default=0.9)
    parser.add_argument('--wavelet-function', dest='wavelet_function', default= None, type=str, help='Method to generate spectrogram')
    parser.add_argument('--sample-wise', dest='sample_wise', default=None)
    parser.add_argument('--output-attention', dest='output_attention', default=None)
    
    return parser
    
        
    