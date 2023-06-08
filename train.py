import os
import argparse
import numpy as np
from pathlib import Path
import time
import torch
import datetime

from torch_geometric.loader import DataLoader

from engine import train_one_epoch
from optim_factory import create_optimizer
from models.model import create_model
from datasets import build_dataset
from configs import get_args_parser
import losses

from models import GAT

def main(args):
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build Dataset
    trainset = build_dataset(args, is_train = True)
    
    valset = build_dataset(args, is_train = False)

    # DataLoader
    train_loader = DataLoader(dataset = trainset,
                              batch_size = args.batch_size,
                              )
    if valset is not None:
        val_loader = DataLoader(dataset = valset,
                                batch_size = int(1.5 * args.batch_size),
                                )
    else:
        val_loader = None

    # Logging Writer
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = None
    
    if args.enable_wandb:
        wandb_writer = None

    # Model
    model = create_model(model_name = args.model,
                         num_classes = args.nb_classes,
                         drop_path_rate = args.drop_path,
                         )

    # Finetune process
    if args.finetune:
        pass

    model.to(device)

    # Criterion Function
    if args.use_focal:
        criterion = losses.FocalLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    
    # Optimizer
    optimizer = create_optimizer(args)

    # Loss Scaler

    # Learning Rate Scheduler
    lr_schedule = None

    max_accuracy = 0.0
    
    # Train process
    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # Set step for logging writer
        if log_writer is not None:
            pass

        if wandb_writer is not None:
            pass
        
        # Train one epoch
        train_stats = train_one_epoch(model = model, criterion= criterion, optimizer = optimizer, data_loader=train_loader,
                                      device = device, update_freq=args.update_freq)

        # Save Model

        # Evaluate validation dataset
        if val_loader is not None:
            pass
        # Logging metric
        
        # Logging checkpoints

    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Graph Source Label', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)