"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def vflip(image, target):
    """Vertical flip for pathology images."""
    flipped_image = F.vflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        # boxes format: [x1, y1, x2, y2]
        # flip y coordinates: new_y = h - old_y
        boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-2)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomVerticalFlip(object):
    """Random vertical flip for pathology images."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


def rotate90(image, target, k):
    """Rotate image and boxes by k*90 degrees counter-clockwise."""
    # Rotate image
    rotated_image = image.rotate(k * 90, expand=False)
    
    w, h = image.size
    
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]  # [x1, y1, x2, y2]
        
        if k == 1:  # 90 degrees CCW
            # (x, y) -> (y, w - x)
            new_boxes = torch.stack([
                boxes[:, 1],           # new x1 = old y1
                w - boxes[:, 2],       # new y1 = w - old x2
                boxes[:, 3],           # new x2 = old y2
                w - boxes[:, 0]        # new y2 = w - old x1
            ], dim=1)
        elif k == 2:  # 180 degrees
            # (x, y) -> (w - x, h - y)
            new_boxes = torch.stack([
                w - boxes[:, 2],       # new x1 = w - old x2
                h - boxes[:, 3],       # new y1 = h - old y2
                w - boxes[:, 0],       # new x2 = w - old x1
                h - boxes[:, 1]        # new y2 = h - old y1
            ], dim=1)
        elif k == 3:  # 270 degrees CCW (90 CW)
            # (x, y) -> (h - y, x)
            new_boxes = torch.stack([
                h - boxes[:, 3],       # new x1 = h - old y2
                boxes[:, 0],           # new y1 = old x1
                h - boxes[:, 1],       # new x2 = h - old y1
                boxes[:, 2]            # new y2 = old x2
            ], dim=1)
        else:
            new_boxes = boxes
        
        target["boxes"] = new_boxes
    
    if "masks" in target:
        target['masks'] = torch.rot90(target['masks'], k, dims=[-2, -1])
    
    return rotated_image, target


class RandomRotation90(object):
    """Random 90-degree rotation for pathology images (0, 90, 180, or 270 degrees)."""
    def __init__(self, p=0.75):
        self.p = p  # Probability of any rotation (vs no rotation)

    def __call__(self, img, target):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            return rotate90(img, target, k)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class RandomColorJitter(object):
    """Random color jitter for pathology images.
    
    Simulates staining variations across different tissue preparations.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5):
        self.p = p
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img, target):
        if random.random() < self.p:
            # ColorJitter expects PIL Image or Tensor, but we need to handle
            # the fact that at this point img is still PIL
            if hasattr(img, 'mode'):  # PIL Image
                img = self.color_jitter(img)
        return img, target


class RandomStainAugmentation(object):
    """
    Stain augmentation for H&E histopathology images.
    
    Simulates variations in Hematoxylin (purple/blue) and Eosin (pink) staining
    that occur due to different lab protocols, tissue preparation, and scanners.
    
    Uses Reinhard-style color transfer in LAB color space to perturb stain colors
    while preserving tissue structure.
    """
    def __init__(self, sigma_l=0.05, sigma_a=0.08, sigma_b=0.08, p=0.5):
        self.sigma_l = sigma_l
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.p = p
    
    def __call__(self, img, target):
        if random.random() < self.p:
            if hasattr(img, 'mode') and img.mode == 'RGB':  # PIL Image
                img = self._augment_stain(img)
        return img, target
    
    def _augment_stain(self, img):
        """Apply stain augmentation in LAB color space."""
        import numpy as np
        from PIL import Image
        
        # Convert to numpy
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # RGB to LAB conversion (simplified approximation)
        # First convert to linear RGB
        img_linear = np.where(img_np <= 0.04045, 
                              img_np / 12.92, 
                              ((img_np + 0.055) / 1.055) ** 2.4)
        
        # RGB to XYZ
        M_rgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = np.dot(img_linear.reshape(-1, 3), M_rgb_to_xyz.T).reshape(img_np.shape)
        
        # XYZ to LAB
        # Reference white point (D65)
        ref_white = np.array([0.95047, 1.0, 1.08883])
        xyz_normalized = xyz / ref_white
        
        epsilon = 0.008856
        kappa = 903.3
        
        f_xyz = np.where(xyz_normalized > epsilon,
                         xyz_normalized ** (1/3),
                         (kappa * xyz_normalized + 16) / 116)
        
        L = 116 * f_xyz[..., 1] - 16
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
        
        # Add random perturbations (the actual stain augmentation)
        L_perturbed = L + np.random.normal(0, self.sigma_l * 100, L.shape)
        a_perturbed = a + np.random.normal(0, self.sigma_a * 128, a.shape)
        b_perturbed = b + np.random.normal(0, self.sigma_b * 128, b.shape)
        
        # Clamp LAB values
        L_perturbed = np.clip(L_perturbed, 0, 100)
        a_perturbed = np.clip(a_perturbed, -128, 127)
        b_perturbed = np.clip(b_perturbed, -128, 127)
        
        # LAB to XYZ
        f_y = (L_perturbed + 16) / 116
        f_x = a_perturbed / 500 + f_y
        f_z = f_y - b_perturbed / 200
        
        x = np.where(f_x ** 3 > epsilon, f_x ** 3, (116 * f_x - 16) / kappa)
        y = np.where(L_perturbed > kappa * epsilon, ((L_perturbed + 16) / 116) ** 3, L_perturbed / kappa)
        z = np.where(f_z ** 3 > epsilon, f_z ** 3, (116 * f_z - 16) / kappa)
        
        xyz_new = np.stack([x * ref_white[0], y * ref_white[1], z * ref_white[2]], axis=-1)
        
        # XYZ to RGB
        M_xyz_to_rgb = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ])
        rgb_linear = np.dot(xyz_new.reshape(-1, 3), M_xyz_to_rgb.T).reshape(img_np.shape)
        rgb_linear = np.clip(rgb_linear, 0, 1)
        
        # Linear RGB to sRGB
        rgb = np.where(rgb_linear <= 0.0031308,
                       12.92 * rgb_linear,
                       1.055 * (rgb_linear ** (1/2.4)) - 0.055)
        
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(rgb)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string