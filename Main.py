import argparse
import yaml
from FlowMatchingRunner import launch

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
  # General Info  
  parser.add_argument('--train', type = str2bool, required = True)
  parser.add_argument('--model', type = str, required = True, help = 'Unet or Transformer ?')
  parser.add_argument('--model_version', type = str, required = True)
  parser.add_argument('--train_mode', type = str, required = True)

  # Learning Rate Scheduler
  parser.add_argument('--learning_rate',        default = 1e-4, type = float, required = False, help = 'learning rate')
  parser.add_argument('--min_learning_rate',    default = 1e-8, type = float, required = False, help = 'init learning rate for warm up')
  parser.add_argument('--lr_scheduler',  default = 'step', type = str, required = False)
  parser.add_argument('--lr_step_size',  default = 150000, type = int , required = False)
  parser.add_argument('--lr_gamma', default = 0.5, type = float, required = False) 
  parser.add_argument('--lr_eta_min', default = 1e-6, type = float, required = False)
  parser.add_argument('--total_epochs', default = 2000, type = int ,required = False)
  parser.add_argument('--warmup_epochs', default = 100, type = int, required = False)


  # Optimizer
  parser.add_argument('--optimizer', default = 'AdamW', type = str, required = False)
  parser.add_argument('--betas', default = (0.9, 0.999), type = tuple, required = False, help = 'ADAM betas')
  parser.add_argument('--max_grad', default = 1.0, type = float, required = False)


 # Dataset 
  parser.add_argument('--dataset', default = 'MNIST', type = str, required = False)
  parser.add_argument('--download', default = True, type = str2bool, required = False, \
                      help = 'Whether to download dataset if missing')
  parser.add_argument('--image_size',           default = 32, type = int,     required = False)
  parser.add_argument('--path',                 default = '',  type = str,    required = False)
  parser.add_argument('--batch_size',           default = 32,  type = int,    required = False)
  parser.add_argument('--num_workers',          default = 0,   type = int,    required = False)
  parser.add_argument('--random_flip',          default = False,  type = str2bool, required = False)
  parser.add_argument('--in_channels',          default = 1, type = int, required = False, help = 'RGB channels')
  parser.add_argument('--out_channels',         default = 1, type = int, required = False, help = 'Channels for output tensor')


  # Unet 
  parser.add_argument('--model_channels', default = 64, type = int, required = False, help = 'Model associated channels')
  parser.add_argument('--num_residual_blocks',  default = 1, type = int, required = False)
  parser.add_argument('--attention_resolution', default = (2, 4, 8), type = tuple, required = False, help = 'Which layers to use attention')
  parser.add_argument('--channel_mult', default = (1, 2, 4, 8), type = tuple, required = False)
  parser.add_argument('--conv_resample', default = True, type = str2bool, required = False) 
  parser.add_argument('--dims', default = 2, type = int, required = False)
  parser.add_argument('--num_attention_heads', default = 8, type = int, required = False)
  parser.add_argument('--use_scale_shift_norm', default = True, type = str2bool, required = False, help = 'Use AdaN ?')
  parser.add_argument('--residual_block_up_down', default = True, type = str2bool, required = False)
  parser.add_argument('--embedding_to_model_dim_ratio', default = 8, type = int, required = False)
  parser.add_argument('--pool_num_attention_head_channel', default = 16, type = int, required = False)
  parser.add_argument('--output_fc_bottleneck_dim', default = 2048, type = int, required = False)
  parser.add_argument('--classifier_pool', default = 'attention', type = str, required = False)

  # Diffusion Transformer
  parser.add_argument('--vae_path', default = 'mse', type = str, required = False, \
                      help = 'KLAutoencoder type, see Hugging Face for more information')
  parser.add_argument('--latent_input_size', default = 32, type = int, required = False)
  parser.add_argument('--patch_size', default = 2, type = int, required = False) 
  parser.add_argument('--transformer_hidden_size', default = 512, type = int, required = False) 
  parser.add_argument('--num_transformer_layers', default = 20, type = int, required = False)
  parser.add_argument('--transformer_num_attention_heads', default = 8, type = int, required = False)
  parser.add_argument('--transformer_ffn_dim', default = 2048, type = int, required = False)
  parser.add_argument('--activation', default = 'GeLU', type = str, required = False)
  parser.add_argument('--learn_sigma', default = False, type = str2bool, required = False)
  parser.add_argument('--frequency_embedding_size', default = 256, type = int, required = False)


  # Timestep distribution
  parser.add_argument('--skewed_timesteps', default = False, type = str2bool, required = False)
  # ODE Sampler
  parser.add_argument('--sampler', default = 'euler', type = str, required = False)
  # Ema helper
  parser.add_argument('--ema_rate', default = 0.99, type = float, required = False)
  
  # Extra training info 
  parser.add_argument('--dropout', default = 0.3, type = float, required = False)
  parser.add_argument('--num_classes', default = None, type = int, required = False)  
  parser.add_argument('--snapshot_freq', default = 10000, type = int, required = False)
  parser.add_argument('--doc', type = str, required = True)
  parser.add_argument('--resume_training', default = False, type = str2bool, required = True)
  parser.add_argument('--device', default = 'cuda', type = str, required = False)
  
  # Sampling info
  parser.add_argument('--num_samples', default = 10, type = int, required = False)
  parser.add_argument('--sampling_batch_size', default = 5, type = int, required = False)
  parser.add_argument('--image_folder',        default = 'samples', type = str, required = False)
  parser.add_argument('--num_sampling_steps', default = 50, type = int, required = False)


  return parser


if __name__ == '__main__':
  parser = get_parser()
  args = parser.parse_args()
  launch(args)
