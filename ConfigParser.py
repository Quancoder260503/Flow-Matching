import argparse
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {'yes', 'true', 't', '1', 'y'}:
        return True
    if lowered in {'no', 'false', 'f', '0', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type = str, required = True)
    parser.add_argument('--train_mode', type = str, required = True)
    parser.add_argument('--dataset', default = 'MNIST', type = str, required = False)
    parser.add_argument('--download', default = True, type = str2bool, required = False, help = 'Whether to download dataset if missing')
    parser.add_argument('--learning_rate',        default = 1e-4, type = float, required = False, help = 'learning rate')
    parser.add_argument('--weight_decay',         default = 1e-5, type = float, required = False, help = 'weight decay')
    parser.add_argument('--betas',                default = (0.9, 0.999), type = tuple, required = False, help = 'ADAM betas')
    parser.add_argument('--image_size',           default = 32, type = int,     required = False)
    parser.add_argument('--path',                 default = '',  type = str,    required = False)
    parser.add_argument('--batch_size',           default = 64,  type = int,    required = False)
    parser.add_argument('--num_workers',          default = 0,   type = int,    required = False)
    parser.add_argument('--optimizer',            default = 'Adam', type = str, required = False)
    parser.add_argument('--random_flip',          default = False,  type = str2bool, required = False)
    parser.add_argument('--in_channels',         default = 1, type = int, required = False, help = 'RGB channels')
    parser.add_argument('--model_channels',       default = 64, type = int, required = False, help = 'Model associated channels')
    parser.add_argument('--out_channels',         default = 1, type = int, required = False, help = 'Channels for output tensor')
    parser.add_argument('--num_residual_blocks',  default = 1, type = int, required = False)
    parser.add_argument('--attention_resolution', default = (2, 4, 8), type = tuple, required = False, help = 'Which layers to use attention')
    parser.add_argument('--dropout', default = 0.1, type = float, required = False)
    parser.add_argument('--channel_mult', default = (1, 2, 4, 8), type = tuple, required = False)
    parser.add_argument('--conv_resample', default = True, type = str2bool, required = False) 
    parser.add_argument('--dims', default = 2, type = int, required = False)
    parser.add_argument('--num_classes', default = None, type = int, required = False)
    parser.add_argument('--num_attention_heads', default = 8, type = int, required = False)
    parser.add_argument('--use_scale_shift_norm', default = True, type = str2bool, required = False, help = 'Use AdaN ?')
    parser.add_argument('--residual_block_up_down', default = True, type = str2bool, required = False)
    parser.add_argument('--embedding_to_model_dim_ratio', default = 8, type = int, required = False)
    parser.add_argument('--pool_num_attention_head_channel', default = 16, type = int, required = False)
    parser.add_argument('--output_fc_bottleneck_dim', default = 2048, type = int, required = False)
    parser.add_argument('--classifier_pool', default = 'attention', type = str, required = False)

    parser.add_argument('--skewed_timesteps', default = False, type = str2bool, required = False)
    parser.add_argument('--sampler', default = 'euler', type = str, required = False)
    
    
    parser.add_argument('--ema_rate', default = 0.99, type = float, required = False)
    parser.add_argument('--noise_scheduler', default = 'cosine', type = str, required = False) 
    parser.add_argument('--model_mean_type', default = 'epsilon', required = False, type = str, choices = ['previous_x', 'start_x', 'epsilon'])
    parser.add_argument('--model_var_type', default = 'fixed_large', required = False, type = str, choices = ['learned', 'fixed_small', 'fixed_large', 'learned_range'])
    parser.add_argument('--loss_type', default = 'mse', required = False, type = str, choices = ['mse', 'rescale_mse', 'kl', 'rescale_kl'])
    parser.add_argument('--rescale_timestep', '--rescale_timesteps', dest = 'rescale_timesteps', default = False, required = False, type = str2bool, help = 'Rescale timesteps during diffusion')
    parser.add_argument('--max_grad', default = 1.0, type = float, required = False)
    parser.add_argument('--snapshot_freq', default = 10000, type = int, required = False)
    parser.add_argument('--doc', type = str, required = True)
    parser.add_argument('--resume_training', default = False, type = str2bool, required = True)
    parser.add_argument('--num_training_epochs', default = 500, type = int ,required = False)
    parser.add_argument('--device', default = 'cuda', type = str, required = False)
    parser.add_argument('--num_samples', default = 10, type = int, required = False)
    parser.add_argument('--sampling_batch_size', default = 5, type = int, required = False)
    parser.add_argument('--image_folder',        default = 'samples', type = str, required = False)
    parser.add_argument('--lr_scheduler',  default = 'step', type = str, required = False)
    parser.add_argument('--lr_step_size',  default = 150000, type = int , required = False)
    parser.add_argument('--lr_gamma', default = 0.5, type = float, required = False) 
    parser.add_argument('--lr_eta_min', default = 1e-6, type = float, required = False)

    parser.add_argument('--num_sampling_steps', default = 200, type = int, required = False)


    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    return args
