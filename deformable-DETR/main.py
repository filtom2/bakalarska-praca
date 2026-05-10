from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import argparse
import datetime
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# Local modules
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import run_training_epoch, run_validation
from models import build_model

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 2e-4
    backbone_lr: float = 2e-5
    projection_lr_mult: float = 0.1
    batch_size: int = 1
    weight_decay: float = 1e-4
    epochs: int = 50
    lr_drop_epoch: int = 40
    gradient_clip: float = 0.1
    use_sgd: bool = False
    seed: int = 42
    num_workers: int = 2


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    backbone: str = 'resnet50'
    use_dilation: bool = False
    position_encoding: str = 'sine'
    position_scale: float = 2 * np.pi
    num_feature_levels: int = 3

    # Transformer
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 128
    feedforward_dim: int = 512
    dropout: float = 0.1
    num_heads: int = 4
    num_queries: int = 5
    encoder_points: int = 4
    decoder_points: int = 4
    feature_height: int = 16
    feature_width: int = 16

    # Loss
    use_auxiliary_loss: bool = True


@dataclass
class LossConfig:
    """Configuration for loss weights and matcher."""
    # Matcher costs
    class_cost: float = 2.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0

    # Loss coefficients
    class_weight: float = 3.0
    bbox_weight: float = 5.0
    giou_weight: float = 2.0
    mask_weight: float = 1.0
    dice_weight: float = 1.0
    focal_alpha: float = 0.4


@dataclass
class DataConfig:
    """Configuration for dataset."""
    dataset_type: str = 'coco'
    data_path: str = './data/coco'
    remove_difficult: bool = False
    cache_images: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    output_dir: str = ''
    device: str = 'cuda'
    resume_checkpoint: str = ''
    start_epoch: int = 0
    eval_only: bool = False


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Deformable DETR Training for Object Detection',
        add_help=True
    )
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--lr', default=2e-4, type=float, help='Base learning rate')
    train_group.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    train_group.add_argument('--lr_backbone', default=2e-5, type=float, help='Backbone learning rate')
    train_group.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    train_group.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    train_group.add_argument('--batch_size', default=1, type=int)
    train_group.add_argument('--weight_decay', default=1e-4, type=float)
    train_group.add_argument('--epochs', default=50, type=int)
    train_group.add_argument('--lr_drop', default=40, type=int)
    train_group.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    train_group.add_argument('--clip_max_norm', default=0.1, type=float, help='Gradient clipping threshold')
    train_group.add_argument('--sgd', action='store_true', help='Use SGD instead of AdamW')
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--backbone', default='resnet50', type=str, help='Backbone network')
    model_group.add_argument('--dilation', action='store_true', help='Use dilated convolutions')
    model_group.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    model_group.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    model_group.add_argument('--num_feature_levels', default=3, type=int)
    model_group.add_argument('--enc_layers', default=3, type=int, help='Encoder layers')
    model_group.add_argument('--dec_layers', default=3, type=int, help='Decoder layers')
    model_group.add_argument('--dim_feedforward', default=512, type=int)
    model_group.add_argument('--hidden_dim', default=128, type=int)
    model_group.add_argument('--dropout', default=0.1, type=float)
    model_group.add_argument('--nheads', default=4, type=int, help='Attention heads')
    model_group.add_argument('--num_queries', default=5, type=int, help='Object queries')
    model_group.add_argument('--dec_n_points', default=4, type=int)
    model_group.add_argument('--enc_n_points', default=4, type=int)
    model_group.add_argument('--last_height', default=16, type=int)
    model_group.add_argument('--last_width', default=16, type=int)
    
    # Loss configuration
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    loss_group.add_argument('--set_cost_class', default=2, type=float)
    loss_group.add_argument('--set_cost_bbox', default=5, type=float)
    loss_group.add_argument('--set_cost_giou', default=2, type=float)
    loss_group.add_argument('--mask_loss_coef', default=1, type=float)
    loss_group.add_argument('--dice_loss_coef', default=1, type=float)
    loss_group.add_argument('--cls_loss_coef', default=3, type=float)
    loss_group.add_argument('--bbox_loss_coef', default=5, type=float)
    loss_group.add_argument('--giou_loss_coef', default=2, type=float)
    loss_group.add_argument('--focal_alpha', default=0.4, type=float)
    
    # Architectural Modifications
    arch_group = parser.add_argument_group('Architectural Modifications')
    arch_group.add_argument('--num_bifpn_layers', default=0, type=int)
    arch_group.add_argument('--no_zoom_in_bounding', action='store_true')
    arch_group.add_argument('--no_zero_bias_init', action='store_true')
    arch_group.add_argument('--scale_factors', nargs='+', type=float)
    arch_group.add_argument('--no_stain_aug', action='store_true')
    arch_group.add_argument('--no_color_aug', action='store_true')

    # Dataset
    data_group = parser.add_argument_group('Dataset')
    data_group.add_argument('--dataset_file', default='mitos')
    data_group.add_argument('--coco_path', default='./data/coco', type=str)
    data_group.add_argument('--remove_difficult', action='store_true')
    data_group.add_argument('--cache_mode', default=False, action='store_true')
    
    # Experiment
    exp_group = parser.add_argument_group('Experiment')
    exp_group.add_argument('--output_dir', default='', help='Output directory')
    exp_group.add_argument('--device', default='cuda')
    exp_group.add_argument('--seed', default=42, type=int)
    exp_group.add_argument('--resume', default='', help='Resume from checkpoint')
    exp_group.add_argument('--start_epoch', default=0, type=int)
    exp_group.add_argument('--eval', action='store_true', help='Evaluation only mode')
    exp_group.add_argument('--num_workers', default=2, type=int)
    
    return parser

def setup_reproducibility(seed: int, rank: int = 0):
    actual_seed = seed + rank
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    random.seed(actual_seed)


def setup_data_loaders(args, dataset_train, dataset_val):
    if args.distributed:
        if args.cache_mode:
            train_sampler = samplers.NodeDistributedSampler(dataset_train)
            val_sampler = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            train_sampler = samplers.DistributedSampler(dataset_train)
            val_sampler = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )
    
    train_loader = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        dataset_val, 
        args.batch_size, 
        sampler=val_sampler,
        drop_last=False, 
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_sampler


def setup_optimizer(model, args):
    """Configure optimizer with parameter groups."""
    def matches_keywords(name, keywords):
        return any(kw in name for kw in keywords)
    
    # Separate parameters into groups with different learning rates
    backbone_params = []
    projection_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if matches_keywords(name, args.lr_backbone_names):
            backbone_params.append(param)
        elif matches_keywords(name, args.lr_linear_proj_names):
            projection_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": projection_params, "lr": args.lr * args.lr_linear_proj_mult},
    ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    return optimizer, scheduler


def load_checkpoint(model, optimizer, scheduler, args):
    """Load model from checkpoint if specified."""
    if not args.resume:
        return args.start_epoch
    
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected = [k for k in unexpected if not (k.endswith('total_params') or k.endswith('total_ops'))]
    
    if missing:
        print(f'Missing keys: {missing}')
    if unexpected:
        print(f'Unexpected keys: {unexpected}')
    
    # Training state if not in eval mode
    if not args.eval and all(k in checkpoint for k in ['optimizer', 'lr_scheduler', 'epoch']):
        import copy
        saved_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Keep current learning rates
        for group, saved in zip(optimizer.param_groups, saved_groups):
            group['lr'] = saved['lr']
            group['initial_lr'] = saved['initial_lr']
        
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scheduler.step_size = args.lr_drop
        scheduler.base_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        scheduler.step(scheduler.last_epoch)
        
        return checkpoint['epoch'] + 1
    
    return args.start_epoch


def save_checkpoint(model, optimizer, scheduler, epoch, args, output_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    
    # Regular checkpoint
    utils.save_on_master(checkpoint, output_dir / 'checkpoint.pth')
    
    # Periodic checkpoint
    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
        utils.save_on_master(checkpoint, output_dir / f'checkpoint{epoch:04}.pth')

# Main
def run_training(args):
    utils.init_distributed_mode(args)
    args.pre_norm = False
    
    device = torch.device(args.device)
    setup_reproducibility(args.seed, utils.get_rank())
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {param_count:,}')
    
    # Setup datasets
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    train_loader, val_loader, train_sampler = setup_data_loaders(args, dataset_train, dataset_val)
    
    # Setup training
    model_for_training = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_for_training = model.module
    
    optimizer, scheduler = setup_optimizer(model_for_training, args)
    
    coco_eval_dataset = get_coco_api_from_dataset(dataset_val)
    output_dir = Path(args.output_dir)
    
    # Load checkpoint
    start_epoch = load_checkpoint(model_for_training, optimizer, scheduler, args)
    
    # Evaluation only mode
    if args.eval:
        stats, evaluator, _ = run_validation(
            model, criterion, postprocessors, val_loader, 
            coco_eval_dataset, device, args.output_dir
        )
        if args.output_dir:
            utils.save_on_master(evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    # Validation after resume
    if args.resume and not args.eval:
        run_validation(model, criterion, postprocessors, val_loader, coco_eval_dataset, device, args.output_dir)
    
    # Training loop
    print(f'\n{"="*60}\nStarting training for {args.epochs} epochs\n{"="*60}')
    training_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = run_training_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm
        )
        scheduler.step()
        
        # Save checkpoint
        if args.output_dir:
            save_checkpoint(model_for_training, optimizer, scheduler, epoch, args, output_dir)
        
        # Validate
        val_stats, evaluator, _ = run_validation(
            model, criterion, postprocessors, val_loader, 
            coco_eval_dataset, device, args.output_dir
        )
        
        # Log epoch summary
        log_data = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'parameters': param_count
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_data) + "\n")
            
            if evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in evaluator.coco_eval:
                    eval_files = ['latest.pth']
                    if epoch % 50 == 0:
                        eval_files.append(f'{epoch:03}.pth')
                    for name in eval_files:
                        torch.save(evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)
    
    # Training complete
    duration = datetime.timedelta(seconds=int(time.time() - training_start))
    print(f'\n{"="*60}\nTraining completed in {duration}\n{"="*60}')

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_training(args)