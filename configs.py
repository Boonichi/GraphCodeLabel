import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Graph Source Label', add_help=False)
    
    # Train parameters
    parser.add_argument('--input_size', default = 224, type = int,
                        help = "Input size of image")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--num_workers', default = 8, type=int,
                        help="Number of worker in DataLoader")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Finetune paramaters:
    parser.add_argument('--finetune', default = None, type = str,
                        help = "Finetuning model with exist checkpoint (best/last)")
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_prefix', default='', type=str)

    # Predict parameters
    parser.add_argument('--test', action = "store_true",
                        help = "Test Process")
    parser.add_argument('--verbose', action = "store_true",
                        help = "Display prediction from model")

    # Model parameters
    parser.add_argument('--model', default="GAT", type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_ver', default = 0, type = int,
                        help ="Number of version of model")
    parser.add_argument('--config', default = None,
                        help = "Add config file that include model params")
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=bool, default=False, help='Using ema to eval during training.')
    
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=0.1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--use_polyloss', action='store_true',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=4e-5, metavar='LR',
                        help='learning rate (default: 4e-5), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Dataset parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--nb_classes", default = 75, type = int,
                        help = "Number of classes in classification")
    parser.add_argument('--data_dir', default='./data/fold_0/train', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_dir', default="./data/fold_0/val", type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default="model_logs",
                        help='path where to tensorboard log')
    parser.add_argument('--disable_eval', type=bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--auto_resume', type=bool, default=True)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_ckpt', type=bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)
    parser.add_argument('--add_general_sink_node', type = bool, default = False)
    parser.add_argument('--add_self_loop', type = bool, default = False)

    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # Verbose parameters
    parser.add_argument("--data_verbose", action = "store_true",
                        help = "Display/Visualize data in data loader")
    
    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', type = str, default = "MacroWoodClassification")
    parser.add_argument('--wandb_key', type = str, default = None,
                        help ="API key of wandb")
    parser.add_argument('--wandb_ckpt', type=bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument("--patience", type = int, default = 5,
                        help="Patience number for early stopping")
    
    
    parser.add_argument('--use_amp', type=bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    
    return parser