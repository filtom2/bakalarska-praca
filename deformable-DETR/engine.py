import math
import sys
from typing import Iterable, Dict, Any, List, Tuple, Optional

import torch
import numpy as np

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher

# Optional WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def create_prediction_visualization(
    image: torch.Tensor,
    predicted_boxes: torch.Tensor,
    confidence_scores: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    epoch_number: int,
    sample_index: int,
    confidence_threshold: float = 0.3
) -> Optional[Any]:
    """
    Generate visualization comparing predictions with ground truth annotations.
    
    Creates a matplotlib figure showing detected objects (red) overlaid with
    ground truth annotations (green) for visual inspection of model performance.
    
    Args:
        image: Input image tensor [C, H, W]
        predicted_boxes: Predicted bounding boxes [N, 4] in xyxy format
        confidence_scores: Prediction confidence values [N]
        ground_truth_boxes: Ground truth boxes [M, 4] in cxcywh normalized format
        epoch_number: Current training epoch
        sample_index: Sample index within batch
        confidence_threshold: Minimum confidence to display prediction
    
    Returns:
        WandB Image object if available, None otherwise
    """
    if not WANDB_AVAILABLE:
        return None
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        from PIL import Image
        
        # Convert tensor to displayable format
        if isinstance(image, torch.Tensor):
            img_array = image.cpu().permute(1, 2, 0).numpy()
            # Denormalize using ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = img_array * std + mean
            img_array = np.clip(img_array, 0, 1)
        else:
            img_array = image
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_array)
        
        height, width = img_array.shape[:2]
        
        # Draw ground truth boxes (green)
        if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
            gt_array = ground_truth_boxes.cpu().numpy()
            for box in gt_array:
                cx, cy, bw, bh = box
                x1 = (cx - bw/2) * width
                y1 = (cy - bh/2) * height
                box_w = bw * width
                box_h = bh * height
                rect = patches.Rectangle(
                    (x1, y1), box_w, box_h,
                    linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth'
                )
                ax.add_patch(rect)
        
        # Draw predictions (red)
        if predicted_boxes is not None and len(predicted_boxes) > 0:
            pred_array = predicted_boxes.cpu().numpy()
            scores_array = confidence_scores.cpu().numpy() if confidence_scores is not None else np.ones(len(predicted_boxes))
            
            for box, score in zip(pred_array, scores_array):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, fontweight='bold')
        
        ax.set_title(f'Epoch {epoch_number} - Sample {sample_index} (Green=GT, Red=Pred)')
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to WandB format
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        result = wandb.Image(pil_image, caption=f"Epoch {epoch_number} - Sample {sample_index}")
        plt.close()
        
        return result
        
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")
        return None


def create_detection_matrix(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    epoch_number: int
) -> Optional[Any]:
    """
    Generate confusion matrix visualization for object detection metrics.
    
    Creates a 2x2 matrix visualization showing TP, FP, FN counts with
    derived precision, recall, and F1 metrics in the title.
    
    Returns:
        WandB Image object if available, None otherwise
    """
    if not WANDB_AVAILABLE:
        return None
    
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Build matrix
        matrix = np.array([
            [true_positives, false_negatives],
            [false_positives, 0]
        ])
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Detected', 'Missed'], fontsize=12)
        ax.set_yticklabels(['Has Object', 'No Object'], fontsize=12)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        
        # Annotate cells
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
        
        plt.colorbar(im, ax=ax)
        
        # Calculate and display metrics
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        ax.set_title(f'Detection Matrix (Epoch {epoch_number})\nP={precision:.3f}, R={recall:.3f}, F1={f1:.3f}', fontsize=14)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        result = wandb.Image(pil_image, caption=f"Detection Matrix - Epoch {epoch_number}")
        plt.close()
        
        return result
        
    except Exception as e:
        print(f"[WARN] Matrix visualization failed: {e}")
        return None


def run_training_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_gradient_norm: float = 0
) -> Dict[str, float]:
    """
    Execute one complete training epoch.
    
    Processes all batches in the data loader, computes losses, and updates
    model parameters using the configured optimizer with optional gradient clipping.
    
    Args:
        model: The detection model to train
        criterion: Loss computation module
        data_loader: Training data iterator
        optimizer: Parameter optimizer
        device: Computation device
        epoch: Current epoch number
        max_gradient_norm: Maximum gradient norm for clipping (0 = no clipping)
    
    Returns:
        Dictionary of training metrics averaged over the epoch
    """
    model.train()
    criterion.train()
    
    metric_tracker = utils.MetricLogger(delimiter="  ")
    metric_tracker.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    log_interval = 50
    
    # Use prefetcher for efficient data loading
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    
    for _ in metric_tracker.log_every(range(len(data_loader)), log_interval, header):
        # Forward pass
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Compute weighted loss
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Aggregate losses across processes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        total_loss_reduced = sum(loss_dict_scaled.values())
        loss_value = total_loss_reduced.item()
        
        # Check for invalid loss
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        if max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        
        optimizer.step()
        
        # Update metrics
        metric_tracker.update(loss=loss_value)
        metric_tracker.update(loss_ce=loss_dict_scaled.get('loss_ce', 0))
        metric_tracker.update(loss_bbox=loss_dict_scaled.get('loss_bbox', 0))
        metric_tracker.update(loss_giou=loss_dict_scaled.get('loss_giou', 0))
        metric_tracker.update(lr=optimizer.param_groups[0]["lr"])
        
        # Get next batch
        samples, targets = prefetcher.next()
    
    # Synchronize metrics
    metric_tracker.synchronize_between_processes()
    print(f"Training stats: {metric_tracker}")
    
    return {k: meter.global_avg for k, meter in metric_tracker.meters.items()}



def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    inter_x1 = max(box1[0].item(), box2[0].item())
    inter_y1 = max(box1[1].item(), box2[1].item())
    inter_x2 = min(box1[2].item(), box2[2].item())
    inter_y2 = min(box1[3].item(), box2[3].item())
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def match_predictions_to_ground_truth(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Match predicted boxes to ground truth using IoU criterion.
    
    Args:
        pred_boxes: Predicted boxes [N, 4] in xyxy format
        gt_boxes: Ground truth boxes [M, 4] in xyxy format
        iou_threshold: Minimum IoU for positive match
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    matched_gt = set()
    true_positives = 0
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            true_positives += 1
    
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt)
    
    return true_positives, false_positives, false_negatives


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: Dict,
    data_loader: Iterable,
    coco_dataset,
    device: torch.device,
    output_dir: str,
    epoch: int = 0,
    max_visualization_samples: int = 8,
    score_threshold: float = 0.35,
    iou_threshold: float = 0.5
) -> Tuple[Dict[str, Any], Optional[CocoEvaluator], List]:
    """
    Execute model validation and compute detection metrics.
    
    Runs inference on the validation set, computes standard COCO metrics,
    and calculates additional detection metrics (precision, recall, F1, F2).
    
    Args:
        model: The detection model to evaluate
        criterion: Loss computation module
        postprocessors: Post-processing modules for outputs
        data_loader: Validation data iterator
        coco_dataset: COCO API dataset for evaluation
        device: Computation device
        output_dir: Directory for saving outputs
        epoch: Current epoch number
        max_visualization_samples: Number of samples to collect for visualization
        score_threshold: Minimum confidence for predictions
        iou_threshold: Minimum IoU for TP/FP classification
    
    Returns:
        Tuple of (metrics_dict, coco_evaluator, visualization_samples)
    """
    model.eval()
    criterion.eval()
    
    metric_tracker = utils.MetricLogger(delimiter="  ")
    metric_tracker.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    evaluator = CocoEvaluator(coco_dataset, iou_types)
    
    # Detection metric accumulators
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Visualization sample collection
    visualization_samples = []
    
    for samples, targets in metric_tracker.log_every(data_loader, 10, 'Validation:'):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Compute losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        metric_tracker.update(
            loss=sum(loss_dict_scaled.values()),
            **loss_dict_scaled
        )
        metric_tracker.update(class_error=loss_dict_reduced['class_error'])
        
        # Post-process outputs
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_sizes)
        
        # Collect visualization samples
        if len(visualization_samples) < max_visualization_samples:
            batch_images = samples.tensors if hasattr(samples, 'tensors') else samples
            
            for i, (target, result) in enumerate(zip(targets, results)):
                if len(visualization_samples) >= max_visualization_samples:
                    break
                if len(target['boxes']) > 0 or len(result['boxes']) > 0:
                    visualization_samples.append((
                        batch_images[i].detach().cpu(),
                        result['boxes'].detach().cpu(),
                        result['scores'].detach().cpu(),
                        target['boxes'].detach().cpu()
                    ))
        
        # Compute TP/FP/FN for each image
        for target, result in zip(targets, results):
            gt_boxes = target['boxes']
            pred_boxes = result['boxes']
            pred_scores = result['scores']
            
            # Filter by confidence
            confident_mask = pred_scores > score_threshold
            pred_boxes = pred_boxes[confident_mask]
            
            # Convert GT from normalized cxcywh to xyxy
            h, w = target['orig_size'].tolist()
            if len(gt_boxes) > 0:
                cx, cy, bw, bh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                gt_xyxy = torch.stack([
                    (cx - bw/2) * w,
                    (cy - bh/2) * h,
                    (cx + bw/2) * w,
                    (cy + bh/2) * h
                ], dim=1)
            else:
                gt_xyxy = torch.zeros((0, 4), device=pred_boxes.device)
            
            # Match predictions to ground truth
            tp, fp, fn = match_predictions_to_ground_truth(pred_boxes, gt_xyxy, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Update COCO evaluator
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if evaluator is not None:
            evaluator.update(res)
    
    # Synchronize and summarize
    metric_tracker.synchronize_between_processes()
    print(f"Validation stats: {metric_tracker}")
    
    if evaluator is not None:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    
    # Compute final metrics
    stats = {k: meter.global_avg for k, meter in metric_tracker.meters.items()}
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    f2_score = 5 * precision * recall / (4 * precision + recall + 1e-6)
    
    stats.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f2_score': f2_score,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    })
    
    print(f"Detection Metrics (IoU≥{iou_threshold}, score≥{score_threshold}):")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, F2={f2_score:.4f}")
    
    if evaluator is not None and 'bbox' in postprocessors.keys():
        stats['coco_eval_bbox'] = evaluator.coco_eval['bbox'].stats.tolist()
    
    return stats, evaluator, visualization_samples

# These aliases maintain compatibility with existing code that imports
# the original function names
train_one_epoch = run_training_epoch
evaluate = run_validation
visualize_predictions_wandb = create_prediction_visualization
create_confusion_matrix_wandb = create_detection_matrix