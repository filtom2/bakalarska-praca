#!/usr/bin/env python3
"""
WSI Patch Extraction for Deformable DETR Training

Extracts patches from Whole Slide Images (WSI) with annotations in COCO format.
Compatible with the Deformable DETR training pipeline.

Sampling strategy:
- 45% real mitosis (class 0)
- 45% mitosis look-alikes (class 1)  
- 10% background tissue

Uses focal loss compatible annotations with two classes for detection.
Category IDs are 0-indexed for Deformable DETR compatibility.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except ImportError:
    print("WARNING: openslide not available. Install with: pip install openslide-python")
    openslide = None


def parse_args():
    parser = argparse.ArgumentParser(description='Extract patches from WSI for DETR training')
    parser.add_argument('--wsi_dir', type=str, required=True,
                        help='Directory containing WSI files')
    parser.add_argument('--annotation_json', type=str, required=True,
                        help='Path to MITOS annotation JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for patches and COCO annotations')
    parser.add_argument('--patches_per_slide', type=int, default=100,
                        help='Number of patches to extract per slide')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of extracted patches (square)')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='Fraction of patches for training (uses upper/lower split)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--level', type=int, default=0,
                        help='WSI pyramid level to extract from (0 = highest resolution)')
    parser.add_argument('--tissue_threshold', type=float, default=0.1,
                        help='Minimum fraction of tissue in patch (0-1)')
    parser.add_argument('--bbox_radius', type=int, default=25,
                        help='Radius for mitosis bounding boxes in pixels')
    # Sampling probabilities
    parser.add_argument('--mitosis_prob', type=float, default=0.45,
                        help='Probability of sampling mitosis patches')
    parser.add_argument('--lookalike_prob', type=float, default=0.45,
                        help='Probability of sampling look-alike patches')
    parser.add_argument('--background_prob', type=float, default=0.10,
                        help='Probability of sampling background patches')
    return parser.parse_args()


def load_annotations(annotation_path: str) -> Dict:
    """
    Load MITOS WSI annotations from JSON or SQLite file.
    
    Supports:
    - JSON format (exported annotations)
    - SQLite format (original MITOS database via SlideRunner)
    """
    print(f"[INFO] Loading annotations from: {annotation_path}")
    
    annotations_by_slide = defaultdict(lambda: {'mitosis': [], 'lookalikes': [], 'hard_negatives': []})
    
    if annotation_path.endswith('.sqlite') or annotation_path.endswith('.db'):
        return load_sqlite_annotations(annotation_path)
    else:
        return load_json_annotations(annotation_path)


def load_sqlite_annotations(db_path: str) -> Dict:
    """Load annotations from SQLite database (SlideRunner format)."""
    import sqlite3
    
    annotations_by_slide = defaultdict(lambda: {'mitosis': [], 'lookalikes': [], 'hard_negatives': []})
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get slides
    slides = cursor.execute("SELECT uid, filename FROM Slides").fetchall()
    
    for slide_uid, filename in slides:
        slide_name = Path(filename).stem
        
        # Get annotations for this slide
        # agreedClass: 1 = mitosis, 2 = look-alike, 7 = hard negative
        annots = cursor.execute("""
            SELECT coordinateX, coordinateY, agreedClass 
            FROM Annotations_coordinates ac
            JOIN Annotations a ON ac.annoId = a.uid
            WHERE a.slide = ?
        """, (slide_uid,)).fetchall()
        
        for x, y, agreed_class in annots:
            bbox_data = {'x': x, 'y': y, 'class': agreed_class}
            
            if agreed_class == 1:
                annotations_by_slide[slide_name]['mitosis'].append(bbox_data)
            elif agreed_class == 2:
                annotations_by_slide[slide_name]['lookalikes'].append(bbox_data)
            elif agreed_class == 7:
                annotations_by_slide[slide_name]['hard_negatives'].append(bbox_data)
    
    conn.close()
    print(f"[INFO] Loaded annotations for {len(annotations_by_slide)} slides from SQLite")
    return dict(annotations_by_slide)


def load_json_annotations(json_path: str) -> Dict:
    """Load annotations from JSON file."""
    annotations_by_slide = defaultdict(lambda: {'mitosis': [], 'lookalikes': [], 'hard_negatives': []})
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle various JSON structures
    if isinstance(data, dict):
        if 'annotations' in data:
            # Format: {"annotations": {...}}
            annots_data = data['annotations']
        elif 'images' in data and 'annotations' in data:
            # COCO-like format
            return load_coco_format_annotations(data)
        else:
            # Direct slide-keyed format
            annots_data = data
    else:
        # List format - likely per-annotation list
        annots_data = data
    
    # Parse slide-keyed annotations
    if isinstance(annots_data, dict):
        for slide_key, slide_data in annots_data.items():
            slide_name = str(slide_key)
            if isinstance(slide_data, dict):
                annotations = slide_data.get('annotations', [])
                if not annotations and 'x' in slide_data:
                    # Single annotation format
                    annotations = [slide_data]
            elif isinstance(slide_data, list):
                annotations = slide_data
            else:
                continue
            
            for annot in annotations:
                x = annot.get('x', annot.get('x1', annot.get('coordinateX', 0)))
                y = annot.get('y', annot.get('y1', annot.get('coordinateY', 0)))
                agreed_class = annot.get('agreedClass', annot.get('class', annot.get('category_id', 0)))
                
                bbox_data = {'x': x, 'y': y, 'class': agreed_class}
                
                if agreed_class == 1:
                    annotations_by_slide[slide_name]['mitosis'].append(bbox_data)
                elif agreed_class == 2:
                    annotations_by_slide[slide_name]['lookalikes'].append(bbox_data)
                elif agreed_class == 7:
                    annotations_by_slide[slide_name]['hard_negatives'].append(bbox_data)
    
    print(f"[INFO] Loaded annotations for {len(annotations_by_slide)} slides from JSON")
    return dict(annotations_by_slide)


def load_coco_format_annotations(data: Dict) -> Dict:
    """Load annotations from COCO format JSON."""
    annotations_by_slide = defaultdict(lambda: {'mitosis': [], 'lookalikes': [], 'hard_negatives': []})
    
    # Build image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    for annot in data['annotations']:
        image_id = annot['image_id']
        filename = id_to_filename.get(image_id, str(image_id))
        slide_name = Path(filename).stem
        
        # COCO bbox: [x, y, width, height]
        bbox = annot['bbox']
        x = bbox[0] + bbox[2] / 2  # center x
        y = bbox[1] + bbox[3] / 2  # center y
        category_id = annot['category_id']
        
        bbox_data = {'x': x, 'y': y, 'class': category_id}
        
        if category_id == 1:
            annotations_by_slide[slide_name]['mitosis'].append(bbox_data)
        elif category_id == 2:
            annotations_by_slide[slide_name]['lookalikes'].append(bbox_data)
        else:
            annotations_by_slide[slide_name]['hard_negatives'].append(bbox_data)
    
    print(f"[INFO] Loaded annotations for {len(annotations_by_slide)} images from COCO format")
    return dict(annotations_by_slide)


def create_tissue_mask(slide: 'openslide.OpenSlide', level: int = -1) -> np.ndarray:
    """
    Generate tissue mask using LAB color thresholding.
    Returns mask at specified level (default: lowest resolution for speed).
    """
    if level < 0:
        level = slide.level_count - 1
    
    # Read at low resolution
    dimensions = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, dimensions)
    img_rgb = np.array(img.convert('RGB'))
    
    # Convert to LAB color space for better tissue detection
    # Simple approach: tissue is not white
    # White background typically has high L and low a, b variance
    gray = np.mean(img_rgb, axis=2)
    
    # Tissue is typically darker than background
    # Also check for color variance (tissue has more color variation)
    color_std = np.std(img_rgb, axis=2)
    
    # Mask: not too bright AND has some color variation
    tissue_mask = (gray < 220) & ((gray > 30) | (color_std > 10))
    
    return tissue_mask.astype(np.uint8)


def check_tissue_overlap(slide: 'openslide.OpenSlide', x: int, y: int, 
                         patch_size: int, level: int, tissue_mask: np.ndarray,
                         mask_level: int, threshold: float = 0.1) -> bool:
    """Check if patch location has sufficient tissue overlap."""
    # Scale coordinates to mask level
    scale = slide.level_downsamples[mask_level] / slide.level_downsamples[level]
    
    mask_x = int(x / slide.level_downsamples[mask_level])
    mask_y = int(y / slide.level_downsamples[mask_level])
    mask_size = max(1, int(patch_size / scale))
    
    # Bounds checking
    mask_h, mask_w = tissue_mask.shape
    x1 = max(0, mask_x)
    y1 = max(0, mask_y)
    x2 = min(mask_w, mask_x + mask_size)
    y2 = min(mask_h, mask_y + mask_size)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    mask_region = tissue_mask[y1:y2, x1:x2]
    tissue_ratio = np.mean(mask_region) if mask_region.size > 0 else 0
    
    return tissue_ratio >= threshold


def extract_patch(slide: 'openslide.OpenSlide', x: int, y: int, 
                  patch_size: int, level: int) -> Optional[Image.Image]:
    """Extract a patch from the slide at given coordinates."""
    try:
        # OpenSlide coordinates are always at level 0
        level_0_x = int(x * slide.level_downsamples[level])
        level_0_y = int(y * slide.level_downsamples[level])
        
        patch = slide.read_region((level_0_x, level_0_y), level, (patch_size, patch_size))
        return patch.convert('RGB')
    except Exception as e:
        print(f"[WARNING] Failed to extract patch at ({x}, {y}): {e}")
        return None


def get_annotations_in_patch(annotations: List[Dict], x: int, y: int, 
                             patch_size: int, bbox_radius: int) -> List[Dict]:
    """
    Get annotations that fall within the patch boundaries.
    Returns bounding boxes in patch-local COCO format [x, y, width, height].
    """
    patch_annotations = []
    d = 2 * bbox_radius  # diameter = width = height of bbox
    
    for annot in annotations:
        annot_x = annot['x']
        annot_y = annot['y']
        
        # Check if annotation center is within patch (with some margin)
        if (x - bbox_radius < annot_x < x + patch_size + bbox_radius and
            y - bbox_radius < annot_y < y + patch_size + bbox_radius):
            
            # Convert to patch-local coordinates
            local_x = annot_x - x - bbox_radius
            local_y = annot_y - y - bbox_radius
            
            # Clip to patch boundaries
            x_min = max(0, local_x)
            y_min = max(0, local_y)
            x_max = min(patch_size, local_x + d)
            y_max = min(patch_size, local_y + d)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Only include if box has reasonable size
            if width > 5 and height > 5:
                # Convert to 0-indexed category: mitosis=0, lookalike=1
                cat_id = 0 if annot['class'] == 1 else 1
                patch_annotations.append({
                    'bbox': [float(x_min), float(y_min), float(width), float(height)],
                    'area': float(width * height),
                    'category_id': cat_id,  # 0 for mitosis, 1 for look-alike
                    'iscrowd': 0
                })
    
    return patch_annotations


def sample_patch_location(slide_width: int, slide_height: int, patch_size: int,
                          annotations: List[Dict], sample_type: str,
                          is_training: bool, jitter_scale: float = 0.5) -> Tuple[int, int]:
    """
    Sample a patch location based on the sampling type.
    
    Args:
        sample_type: 'mitosis', 'lookalike', or 'background'
        is_training: If True, sample from upper half; else lower half
    """
    half_height = slide_height // 2
    
    if sample_type in ['mitosis', 'lookalike']:
        if not annotations:
            # Fallback to random tissue location
            return sample_random_location(slide_width, slide_height, patch_size, is_training)
        
        # Filter annotations by train/val region
        valid_annots = []
        for annot in annotations:
            if is_training and annot['y'] < half_height:
                valid_annots.append(annot)
            elif not is_training and annot['y'] >= half_height:
                valid_annots.append(annot)
        
        if not valid_annots:
            return sample_random_location(slide_width, slide_height, patch_size, is_training)
        
        # Pick random annotation and add jitter
        annot = random.choice(valid_annots)
        jitter_x = int(random.uniform(-patch_size * jitter_scale, patch_size * jitter_scale))
        jitter_y = int(random.uniform(-patch_size * jitter_scale, patch_size * jitter_scale))
        
        x = annot['x'] - patch_size // 2 + jitter_x
        y = annot['y'] - patch_size // 2 + jitter_y
        
        # Clamp to slide boundaries
        x = max(0, min(x, slide_width - patch_size))
        y = max(0, min(y, slide_height - patch_size))
        
        return x, y
    
    else:  # background
        return sample_random_location(slide_width, slide_height, patch_size, is_training)


def sample_random_location(slide_width: int, slide_height: int, 
                          patch_size: int, is_training: bool) -> Tuple[int, int]:
    """Sample random location from appropriate half of slide."""
    half_height = slide_height // 2
    
    x = random.randint(0, max(0, slide_width - patch_size))
    
    if is_training:
        y = random.randint(0, max(0, half_height - patch_size))
    else:
        y = random.randint(half_height, max(half_height, slide_height - patch_size))
    
    return x, y


def process_slide(slide_path: str, slide_annotations: Dict, args, 
                  image_id_start: int, annot_id_start: int,
                  train_images: List, train_annotations: List,
                  val_images: List, val_annotations: List) -> Tuple[int, int]:
    """
    Process a single slide and extract patches.
    Returns updated image_id and annotation_id counters.
    """
    slide_name = Path(slide_path).stem
    print(f"[INFO] Processing slide: {slide_name}")
    
    try:
        slide = openslide.open_slide(slide_path)
    except Exception as e:
        print(f"[ERROR] Failed to open slide {slide_path}: {e}")
        return image_id_start, annot_id_start
    
    level = min(args.level, slide.level_count - 1)
    slide_width, slide_height = slide.level_dimensions[level]
    down_factor = slide.level_downsamples[level]
    
    # Generate tissue mask at lowest resolution
    mask_level = slide.level_count - 1
    tissue_mask = create_tissue_mask(slide, mask_level)
    
    # Collect all annotations for this slide, scaled to current level
    all_mitosis = []
    all_lookalikes = []
    
    for annot in slide_annotations.get('mitosis', []):
        all_mitosis.append({
            'x': int(annot['x'] / down_factor),
            'y': int(annot['y'] / down_factor),
            'class': 1
        })
    
    for annot in slide_annotations.get('lookalikes', []):
        all_lookalikes.append({
            'x': int(annot['x'] / down_factor),
            'y': int(annot['y'] / down_factor),
            'class': 2
        })
    
    # Also use hard negatives as additional look-alikes for sampling
    for annot in slide_annotations.get('hard_negatives', []):
        all_lookalikes.append({
            'x': int(annot['x'] / down_factor),
            'y': int(annot['y'] / down_factor),
            'class': 2  # Treat as look-alike
        })
    
    all_annotations = all_mitosis + all_lookalikes
    
    print(f"  - Slide dimensions at level {level}: {slide_width}x{slide_height}")
    print(f"  - Mitosis annotations: {len(all_mitosis)}")
    print(f"  - Look-alike annotations: {len(all_lookalikes)}")
    
    # Calculate patches per split
    train_patches = int(args.patches_per_slide * args.train_val_split)
    val_patches = args.patches_per_slide - train_patches
    
    # Sampling probabilities
    sample_probs = [args.mitosis_prob, args.lookalike_prob, args.background_prob]
    sample_types = ['mitosis', 'lookalike', 'background']
    
    image_id = image_id_start
    annot_id = annot_id_start
    
    for is_training, num_patches, img_list, annot_list in [
        (True, train_patches, train_images, train_annotations),
        (False, val_patches, val_images, val_annotations)
    ]:
        split_name = 'train' if is_training else 'val'
        extracted = 0
        attempts = 0
        max_attempts = num_patches * 10
        
        while extracted < num_patches and attempts < max_attempts:
            attempts += 1
            
            # Sample type based on probabilities
            sample_type = random.choices(sample_types, sample_probs)[0]
            
            # Choose annotations based on sample type
            if sample_type == 'mitosis':
                annots_for_sampling = all_mitosis
            elif sample_type == 'lookalike':
                annots_for_sampling = all_lookalikes
            else:
                annots_for_sampling = []
            
            # Get patch location
            x, y = sample_patch_location(
                slide_width, slide_height, args.patch_size,
                annots_for_sampling, sample_type, is_training
            )
            
            # Check tissue overlap
            if not check_tissue_overlap(slide, x, y, args.patch_size, level, 
                                       tissue_mask, mask_level, args.tissue_threshold):
                continue
            
            # Extract patch
            patch = extract_patch(slide, x, y, args.patch_size, level)
            if patch is None:
                continue
            
            # Get annotations in this patch
            patch_annots = get_annotations_in_patch(
                all_annotations, x, y, args.patch_size, 
                int(args.bbox_radius / down_factor)
            )
            
            # Save patch
            patch_filename = f"{slide_name}_{split_name}_{image_id:06d}.png"
            patch_dir = 'train2017' if is_training else 'val2017'
            patch_path = Path(args.output_dir) / patch_dir / patch_filename
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch.save(patch_path)
            
            # Add to COCO format
            img_list.append({
                'id': image_id,
                'file_name': patch_filename,
                'width': args.patch_size,
                'height': args.patch_size
            })
            
            for annot in patch_annots:
                annot['id'] = annot_id
                annot['image_id'] = image_id
                annot_list.append(annot)
                annot_id += 1
            
            image_id += 1
            extracted += 1
        
        print(f"  - Extracted {extracted} {split_name} patches")
    
    slide.close()
    return image_id, annot_id


def create_coco_categories() -> List[Dict]:
    """Create COCO category definitions for mitosis detection (0-indexed)."""
    return [
        {'id': 0, 'name': 'mitosis', 'supercategory': 'cell'},
        {'id': 1, 'name': 'lookalike', 'supercategory': 'cell'}
    ]


def save_coco_annotations(output_dir: str, images: List[Dict], 
                          annotations: List[Dict], split: str):
    """Save annotations in COCO format."""
    ann_dir = Path(output_dir) / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': create_coco_categories()
    }
    
    ann_file = ann_dir / f'instances_{split}2017.json'
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"[INFO] Saved {len(images)} images and {len(annotations)} annotations to {ann_file}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("WSI Patch Extraction for Deformable DETR")
    print("=" * 60)
    print(f"WSI Directory: {args.wsi_dir}")
    print(f"Annotation JSON: {args.annotation_json}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Patches per slide: {args.patches_per_slide}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Sampling: {args.mitosis_prob*100:.0f}% mitosis, "
          f"{args.lookalike_prob*100:.0f}% look-alikes, "
          f"{args.background_prob*100:.0f}% background")
    print("=" * 60)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load annotations
    annotations_by_slide = load_annotations(args.annotation_json)
    
    # Find WSI files
    wsi_dir = Path(args.wsi_dir)
    wsi_extensions = ['.svs', '.ndpi', '.tiff', '.tif', '.mrxs', '.scn']
    wsi_files = []
    for ext in wsi_extensions:
        wsi_files.extend(wsi_dir.glob(f'*{ext}'))
        wsi_files.extend(wsi_dir.glob(f'*{ext.upper()}'))
    
    print(f"[INFO] Found {len(wsi_files)} WSI files")
    
    if not wsi_files:
        print("[ERROR] No WSI files found!")
        return
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'train2017').mkdir(exist_ok=True)
    (Path(args.output_dir) / 'val2017').mkdir(exist_ok=True)
    
    # Process slides
    train_images, train_annotations = [], []
    val_images, val_annotations = [], []
    image_id, annot_id = 1, 1
    
    for wsi_path in tqdm(wsi_files, desc="Processing slides"):
        slide_name = wsi_path.stem
        
        # Find matching annotations
        slide_annots = {}
        for key in annotations_by_slide:
            if slide_name in str(key) or str(key) in slide_name:
                slide_annots = annotations_by_slide[key]
                break
        
        if not slide_annots:
            print(f"[WARNING] No annotations found for slide: {slide_name}")
            # Still extract background patches
            slide_annots = {'mitosis': [], 'lookalikes': [], 'hard_negatives': []}
        
        image_id, annot_id = process_slide(
            str(wsi_path), slide_annots, args,
            image_id, annot_id,
            train_images, train_annotations,
            val_images, val_annotations
        )
    
    # Save COCO annotations
    save_coco_annotations(args.output_dir, train_images, train_annotations, 'train')
    save_coco_annotations(args.output_dir, val_images, val_annotations, 'val')
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"Training: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Validation: {len(val_images)} images, {len(val_annotations)} annotations")
    print("=" * 60)


if __name__ == '__main__':
    main()
