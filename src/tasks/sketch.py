"""Preprocessing methods specific to the sketch-to-image task."""

import random
import numpy as np
import warnings
from PIL import Image
from functools import partial
from controlnet_aux import LineartDetector
from . import base_transform

# Silly warning that keeps coming up when we import lineart detector.
warnings.filterwarnings("ignore", "Overwriting tiny_vit")
    
def detect_sketch(img, detector):    
    # This is based on qualitative eval of the sketch detectors. The resolutions
    # are different per style but they roughly reflect the same amount of transition
    # from coarse to fine-grained.
    resolutions = [128, 256, 384, 512]
    c_res = np.random.choice(resolutions)
    out = detector(img, detect_resolution=c_res, image_resolution=512)
    out = out.resize(img.size)
    out = np.array(out.convert("L"))
    out = Image.fromarray(out).convert("RGB")

    return out

def add_transform_to_dataset(dataset, resolution, accelerator):
    """Add the necessary transforms to the dataset."""

    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    cond_fn = partial(detect_sketch, detector=detector)
    transform_fn = partial(
        base_transform, cond_callable=cond_fn, resolution=resolution
    )

    if accelerator is not None:
        with accelerator.main_process_first():
            dataset = dataset.with_transform(transform_fn)
    else:
        dataset = dataset.with_transform(transform_fn)

    return dataset