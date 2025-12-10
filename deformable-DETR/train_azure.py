"""
Azure ML Training Script for Deformable-DETR Mitosis Detection
Optimized for single V100 GPU with 134,000 images
Includes mitosis-specific optimizations for 256x256 patches with 1-2 objects
"""
import argparse
import datetime
import json
import math
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate
from models import build_model

import wandb



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR for Mitosis Detection', add_help=False)
    
    # Azure ML paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Azure ML dataset (contains train2017, val2017, annotations)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save outputs (checkpoints, logs)')
    
    # Training hyperparameters - optimized for mitosis detection
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (V100 can handle 8-16)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='More epochs for medical imaging')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Conservative LR for medical images')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='Backbone learning rate (10x lower)')
    parser.add_argument('--lr_drop', default=80, type=int,
                        help='LR drop epoch (later for longer training)')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--use_balanced_sampling', action='store_true',
                        help='Use balanced sampling (2x weight for positive patches)')
    
    # Model architecture - optimized for 256x256 patches with 1-2 objects
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Backbone architecture')
    parser.add_argument('--num_queries', default=5, type=int,
                        help='Number of query slots (3-5 optimal for 1-2 objects)')
    parser.add_argument('--num_feature_levels', default=3, type=int,
                        help='Feature levels (3 sufficient for 256x256 patches)')
    
    # Transformer - smaller model for simpler task
    parser.add_argument('--enc_layers', default=3, type=int,
                        help='Number of encoding layers')
    parser.add_argument('--dec_layers', default=3, type=int,
                        help='Number of decoding layers')
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help='FFN dimension (smaller for 256x256 patches)')
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help='Transformer dimension (smaller model)')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=4, type=int,
                        help='Number of attention heads')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--last_height', default=16, type=int)
    parser.add_argument('--last_width', default=16, type=int)
    
    # Position embedding
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    
    # Loss
    parser.add_argument('--aux_loss', action='store_true', default=True,
                        help='Auxiliary decoding losses')
    
    # Matcher - rebalanced for mitosis (localization more important)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help='Reduced - classification easy with few objects')
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=3, type=float,
                        help='Increased for precise localization')
    
    # Loss coefficients - rebalanced for mitosis detection
    parser.add_argument('--cls_loss_coef', default=1, type=float,
                        help='Reduced - classification easy with 1-2 objects')
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=3, type=float,
                        help='Increased for precise localization')
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Dataset
    parser.add_argument('--dataset_file', default='mitos', type=str)
    
    # Device
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=42, type=int)
    
    # Fixed parameters
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], 
                        type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--cache_mode', default=False, action='store_true')
    
    return parser


def create_balanced_sampler(dataset):
    """
    Create weighted sampler that upweights patches containing mitoses.
    Patches with objects get 2x weight vs empty patches.
    """
    print("[INFO] Creating balanced sampler...")
    weights = []
    for idx in range(len(dataset)):
        ann_ids = dataset.coco.getAnnIds(imgIds=dataset.ids[idx])
        # 2x weight for patches with mitoses
        weights.append(2.0 if len(ann_ids) > 0 else 1.0)
    
    positive_count = sum(1 for w in weights if w > 1.0)
    print(f"[INFO] Balanced sampler: {positive_count} positive, {len(weights)-positive_count} negative patches")
    return WeightedRandomSampler(weights, len(weights))


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Warmup + cosine annealing scheduler.
    - First warmup_epochs: linear warmup from 0 to base LR
    - Remaining epochs: cosine decay to near 0
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args):
    run_name = os.getenv('WANDB_RUN_NAME', f'deformable-detr-e{args.epochs}-bs{args.batch_size}')
    wandb.init(
        project=os.getenv('WANDB_PROJECT', 'Mitos_BP_DeformableDETR'),
        name=run_name,
        config=vars(args)
        )
    
    print("="*80)
    print("Deformable-DETR for Mitosis Detection")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Num queries: {args.num_queries}")
    print(f"Encoder layers: {args.enc_layers}, Decoder layers: {args.dec_layers}")
    print("="*80)
    
    device = torch.device(args.device)
    args.pre_norm = False
    
    # Fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build model
    print("\n[INFO] Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Number of trainable parameters: {n_parameters:,}')
    
    # Build datasets
    print(f"\n[INFO] Loading datasets from {args.data_path}...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print(f"[INFO] Train size: {len(dataset_train)}, Val size: {len(dataset_val)}")
    
    # Data loaders with optional balanced sampling
    if args.use_balanced_sampling:
        sampler_train = create_balanced_sampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val, 
        args.batch_size, 
        sampler=sampler_val,
        drop_last=False, 
        collate_fn=utils.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Optimizer with different LR for backbone
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not match_name_keywords(n, args.lr_backbone_names) 
                      and not match_name_keywords(n, args.lr_linear_proj_names) 
                      and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine scheduler for stable training
    lr_scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    print(f"[INFO] Using warmup ({args.warmup_epochs} epochs) + cosine annealing scheduler")
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get COCO evaluator
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    start_time = time.time()
    best_map = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        
        # Save checkpoint
        checkpoint_path = output_dir / 'checkpoint.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)
        
        # Evaluate
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, str(output_dir)
        )
        
        # Extract metrics
        map_50 = test_stats.get('coco_eval_bbox', [0]*12)[1]  # AP@50
        map_75 = test_stats.get('coco_eval_bbox', [0]*12)[2]  # AP@75
        map_avg = test_stats.get('coco_eval_bbox', [0]*12)[0]  # AP@[.5:.95]
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {train_stats['loss']:.4f}")
        print(f"[Epoch {epoch+1}] Val mAP@50: {map_50:.4f}, mAP@75: {map_75:.4f}, mAP: {map_avg:.4f}")
        
        # Log to WandB - clean metrics only
        log_dict = {
            "epoch": epoch + 1,
            # Training
            "train/loss": train_stats['loss'],
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/loss_ce": train_stats.get('loss_ce', 0),
            "train/loss_bbox": train_stats.get('loss_bbox', 0),
            "train/loss_giou": train_stats.get('loss_giou', 0),
            # Validation - mAP metrics
            "val/mAP": map_avg,
            "val/mAP_50": map_50,
            "val/mAP_75": map_75,
            # Validation - precision/recall/F-scores
            "val/precision": test_stats.get('precision', 0),
            "val/recall": test_stats.get('recall', 0),
            "val/f1_score": test_stats.get('f1_score', 0),
            "val/f2_score": test_stats.get('f2_score', 0),
            # Confusion matrix components
            "val/TP": test_stats.get('tp', 0),
            "val/FP": test_stats.get('fp', 0),
            "val/FN": test_stats.get('fn', 0),
        }
        
        wandb.log(log_dict)
        
        # Save best model based on mAP@50
        if map_50 > best_map:
            best_map = map_50
            best_checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'map_50': map_50,
                'map_avg': map_avg,
            }, best_checkpoint_path)
            print(f"[INFO] ✓ Best model saved! mAP@50: {best_map:.4f}")
        
        # Save log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\n' + "="*80)
    print(f'Training completed in {total_time_str}')
    print(f'Best mAP@50: {best_map:.4f}')
    print("="*80)
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training for mitosis detection',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
