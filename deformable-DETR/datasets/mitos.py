"""
MITOS dataset for mitosis detection using extracted patches in COCO format.
"""
from pathlib import Path
import torch
import torch.utils.data
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from .torchvision_datasets import CocoDetection as TvCocoDetection
from .coco import ConvertCocoPolysToMask


class MitosDetection(TvCocoDetection):
    """MITOS dataset in COCO format for mitosis detection."""
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, 
                 cache_mode=False, local_rank=0, local_size=1):
        super(MitosDetection, self).__init__(
            img_folder, ann_file,
            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(MitosDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def make_mitos_transforms(image_set):
    """
    Transforms for MITOS dataset.
    Uses fixed 256x256 patches so less aggressive augmentation.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResize([256], max_size=512),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([256], max_size=512),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    """
    Build MITOS dataset for mitosis detection.
    
    Args:
        image_set: 'train' or 'val'
        args: must contain args.data_path pointing to extracted patch dataset
    """
    root = Path(args.data_path)
    
    PATHS = {
        "train": (root / "train2017", root / "annotations" / "instances_train2017.json"),
        "val": (root / "val2017", root / "annotations" / "instances_val2017.json"),
    }
    
    img_folder, ann_file = PATHS[image_set]
    
    print(f"[MITOS] Loading {image_set} from:")
    print(f"  Images: {img_folder}")
    print(f"  Annotations: {ann_file}")
    
    dataset = MitosDetection(
        str(img_folder), 
        str(ann_file), 
        transforms=make_mitos_transforms(image_set), 
        return_masks=False,
        local_rank=get_local_rank(), 
        local_size=get_local_size()
    )
    
    return dataset
