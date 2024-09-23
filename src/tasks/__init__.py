from torch import nn
from torchvision import transforms
from functools import partial
from typing import List, Dict, Callable, Union
from ..util_pil import (clean_image,
                        replace_alpha_with_white)
from ..util import ensure_spatial_dims_div

def get_crop_params(img, tgt_size):             
    y1 = max(0, int(round((img.height - tgt_size) / 2.0)))
    x1 = max(0, int(round((img.width - tgt_size) / 2.0)))
    return y1, x1

def base_transform(examples: List[Dict], 
                   resolution: Union[int, None],
                   cond_callable: Callable):
    """
    This is a transformation template which is intended to be passed
    into dataset.with_transform(). However, this particular method
    takes more than just `examples`, and so you will need to curry
    the function before-hand to fill in the right-most arguments,
    e.g:

    ```
    my_transform = partial(transform_template, **kwargs)
    dataset.with_transform(my_transform)
    ```    
    """

    if resolution is not None:
        resize_smallest_edge = transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR
        )
        center_crop = transforms.CenterCrop(resolution)
    else:
        resize_smallest_edge = nn.Identity()
        center_crop = nn.Identity()

    # NOTE: only defined for torch tensors, not PIL
    ensure_div = partial(ensure_spatial_dims_div, n=16)
    
    to_tensor = transforms.ToTensor()
    to_center = transforms.Normalize([0.5], [0.5])

    # Clean images here. If they are RGBA and have dirty pixel values (see docstring
    # of clean_image) we fix it here. Either way, everything gets converted to RGB.
    images = [clean_image(image) for image in examples["image_file"]]  # RGB guaranteed
    # 0b. Resize smaller edge, if doing fixed res training.
    images = [resize_smallest_edge(image) for image in images]

    # For SDXL micro-conditioning.
    img_params = [(img.height, img.width) for img in images ]
    if resolution is not None:
        crop_params = [ get_crop_params(img, resolution) for img in images ]
    else:
        crop_params = [ (0,0) for _ in images ]
    # Is a no-op if we do variable res training
    images = [center_crop(img) for img in images]

    # ------------------

    # 1. Cast to Torch tensor.
    # 2. Ensure spatial dims are divisible by 16.
    # 3. Center in [-1, 1].
    images_torch = [to_center(ensure_div(to_tensor(img))) for img in images]

    # ------------------

    # 1. Construct the conditioning image in real-time.
    cond_images = [
        cond_callable(image).resize(image.size) for image in images
    ]
    # 2. Cast to Torch Tensor
    # 3. Ensure spatial dims are divisible by 16.
    # 4. Center image.
    cond_images_torch = [
        to_center(ensure_div(to_tensor(image))) for image in cond_images
    ]

    examples["pixel_values"] = images_torch
    examples["crop_params"] = crop_params
    examples["img_params"] = img_params
    examples["conditioning_images"] = cond_images
    examples["conditioning_pixel_values"] = cond_images_torch

    return examples