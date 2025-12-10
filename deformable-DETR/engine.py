
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher

import wandb
import numpy as np


def visualize_predictions_wandb(image, pred_boxes, pred_scores, gt_boxes, epoch, batch_idx, max_images=16):
    """
    Visualize predictions vs ground truth and return WandB Image object.
    
    Args:
        image: Tensor [C, H, W] or numpy array
        pred_boxes: Tensor [N, 4] in xyxy format
        pred_scores: Tensor [N,] confidence scores
        gt_boxes: Tensor [M, 4] in cxcywh normalized format
        epoch: Current epoch number
        batch_idx: Batch index
        max_images: Maximum number of images to log per epoch
        
    Returns:
        wandb.Image object or None if failed
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        from PIL import Image
        
        # Convert image tensor to numpy
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = image
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        
        h, w = img_np.shape[:2]
        
        # Ground truth in green
        if gt_boxes is not None and len(gt_boxes) > 0:
            gt_boxes_np = gt_boxes.cpu().numpy()
            for box in gt_boxes_np:
                # Convert cxcywh normalized to xyxy pixels
                cx, cy, bw, bh = box
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                box_w = bw * w
                box_h = bh * h
                rect = patches.Rectangle((x1, y1), box_w, box_h,
                                         linewidth=3, edgecolor='lime', facecolor='none',
                                         label='GT')
                ax.add_patch(rect)
        
        # Predictions in red (with score > 0.3)
        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_boxes_np = pred_boxes.cpu().numpy()
            pred_scores_np = pred_scores.cpu().numpy() if pred_scores is not None else np.ones(len(pred_boxes))
            for box, score in zip(pred_boxes_np, pred_scores_np):
                if score > 0.3:  # Only show confident predictions
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, fontweight='bold')
        
        ax.set_title(f'Epoch {epoch} - Batch {batch_idx} (Green=GT, Red=Pred)', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to WandB Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        wandb_image = wandb.Image(pil_image, caption=f"Epoch {epoch} - Batch {batch_idx}")
        plt.close()
        
        return wandb_image
        
    except Exception as e:
        print(f"[WARN] WandB visualization failed: {e}")
        return None


def create_confusion_matrix_wandb(tp, fp, fn, epoch):
    """
    Create a confusion matrix visualization for WandB.
    
    For object detection, we have:
    - TP: Correctly detected objects
    - FP: False detections (predicted but no GT)
    - FN: Missed objects (GT but no prediction)
    - TN: Not applicable for object detection (would be infinite background patches)
    
    Args:
        tp: True positives count
        fp: False positives count
        fn: False negatives count
        epoch: Current epoch number
        
    Returns:
        wandb.Table with confusion matrix data
    """
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
        
        # Create confusion matrix style visualization
        # For detection: rows = Actual, cols = Predicted
        # [TP, FN]
        # [FP, --]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create matrix data
        matrix = np.array([[tp, fn], [fp, 0]])
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Detected', 'Not Detected'], fontsize=12)
        ax.set_yticklabels(['Has Object', 'No Object'], fontsize=12)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        
        # Add text annotations
        labels = [['TP', 'FN'], ['FP', 'N/A']]
        for i in range(2):
            for j in range(2):
                if i == 1 and j == 1:
                    text = 'N/A'
                    color = 'gray'
                else:
                    text = f'{labels[i][j]}\n{matrix[i, j]:,}'
                    color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=14, color=color, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Title with metrics
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        ax.set_title(f'Detection Matrix (Epoch {epoch})\nP={precision:.3f}, R={recall:.3f}, F1={f1:.3f}', fontsize=14)
        
        plt.tight_layout()
        
        # Convert to WandB Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        wandb_image = wandb.Image(pil_image, caption=f"Confusion Matrix - Epoch {epoch}")
        plt.close()
        
        return wandb_image
        
    except Exception as e:
        print(f"[WARN] Confusion matrix visualization failed: {e}")
        return None


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_ce=loss_dict_reduced_scaled.get('loss_ce', 0))
        metric_logger.update(loss_bbox=loss_dict_reduced_scaled.get('loss_bbox', 0))
        metric_logger.update(loss_giou=loss_dict_reduced_scaled.get('loss_giou', 0))
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch=0, max_vis_samples=8):
    """
    Evaluate the model and collect metrics.
    
    Args:
        model: The model to evaluate
        criterion: Loss criterion
        postprocessors: Post-processing modules
        data_loader: Validation data loader
        base_ds: Base COCO dataset for evaluation
        device: Device to run on
        output_dir: Output directory
        epoch: Current epoch number for logging
        max_vis_samples: Maximum visualization samples to collect
        
    Returns:
        stats: Dictionary of metrics
        coco_evaluator: COCO evaluator object
        visualization_samples: List of (image, pred_boxes, pred_scores, gt_boxes) tuples
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    # Track predictions vs ground truth for precision/recall
    all_tp = 0
    all_fp = 0
    all_fn = 0  # (missed detections)
    iou_threshold = 0.5
    score_threshold = 0.3
    
    # Collect visualization samples
    visualization_samples = []
    batch_idx = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # Collect visualization samples (first few batches with objects)
        if len(visualization_samples) < max_vis_samples:
            # Get the raw images from samples (NestedTensor)
            if hasattr(samples, 'tensors'):
                batch_images = samples.tensors
            else:
                batch_images = samples
            
            for i, (target, result) in enumerate(zip(targets, results)):
                if len(visualization_samples) >= max_vis_samples:
                    break
                # Prefer images with ground truth objects
                if len(target['boxes']) > 0 or len(result['boxes']) > 0:
                    visualization_samples.append((
                        batch_images[i].detach().cpu(),  # Image tensor
                        result['boxes'].detach().cpu(),   # Predicted boxes (xyxy)
                        result['scores'].detach().cpu(),  # Prediction scores
                        target['boxes'].detach().cpu()    # GT boxes (cxcywh normalized)
                    ))
        
        batch_idx += 1
        
        # Compute TP, FP, FN for precision/recall
        for target, result in zip(targets, results):
            gt_boxes = target['boxes']  # Ground truth boxes (cxcywh normalized)
            pred_boxes = result['boxes']  # Predicted boxes (xyxy)
            pred_scores = result['scores']
            
            # Filter predictions by score threshold
            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            
            # Convert GT from cxcywh normalized to xyxy
            h, w = target['orig_size'].tolist()
            if len(gt_boxes) > 0:
                gt_cx, gt_cy, gt_w, gt_h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                gt_x1 = (gt_cx - gt_w / 2) * w
                gt_y1 = (gt_cy - gt_h / 2) * h
                gt_x2 = (gt_cx + gt_w / 2) * w
                gt_y2 = (gt_cy + gt_h / 2) * h
                gt_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)
            else:
                gt_xyxy = torch.zeros((0, 4), device=pred_boxes.device)
            
            # Match predictions to GT using IoU
            matched_gt = set()
            tp = 0
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_xyxy):
                    if gt_idx in matched_gt:
                        continue
                    # Compute IoU
                    inter_x1 = max(pred_box[0].item(), gt_box[0].item())
                    inter_y1 = max(pred_box[1].item(), gt_box[1].item())
                    inter_x2 = min(pred_box[2].item(), gt_box[2].item())
                    inter_y2 = min(pred_box[3].item(), gt_box[3].item())
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union_area = pred_area + gt_area - inter_area
                    iou = inter_area / (union_area + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    tp += 1
            
            fp = len(pred_boxes) - tp
            fn = len(gt_xyxy) - len(matched_gt)
            
            all_tp += tp
            all_fp += fp
            all_fn += fn

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
   
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Compute precision, recall, F1, F2
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    f2_score = 5 * precision * recall / (4 * precision + recall + 1e-6)
    
    stats['precision'] = precision
    stats['recall'] = recall
    stats['f1_score'] = f1_score
    stats['f2_score'] = f2_score
    stats['tp'] = all_tp
    stats['fp'] = all_fp
    stats['fn'] = all_fn
    
    print(f"Detection Metrics (IoU≥{iou_threshold}, score≥{score_threshold}):")
    print(f"  TP={all_tp}, FP={all_fp}, FN={all_fn}")
    print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, F2={f2_score:.4f}")
    
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator, visualization_samples