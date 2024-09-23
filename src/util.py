import torch
import os
import shutil
import diffusers
import numpy as np
from time import time
from torch.nn import functional as F
from PIL import Image, ImageOps
from typing import List, Tuple, Dict
from controlnet_aux import LineartDetector
from .util_pil import make_alpha_all_ones


def check_nan_weights(module, identifier):
    for param_name, param in module.named_parameters():
        if torch.isnan(param.data).any().item():
            raise ValueError(f"found nan weights in {identifier} : {param_name}")

def get_closest_divisible_by_n(a, n):
    if a % n == 0:
        return a
    b = a + (n - (a % n))
    return b

def ensure_spatial_dims_div(img: torch.Tensor, n: int):
    """Ensure the image's spatial dims are divisible by n.
    Return image which satisfies this constraint."""
    height, width = img.size(1), img.size(2)
    new_height = get_closest_divisible_by_n(height, n)
    new_width = get_closest_divisible_by_n(width, n)
    return F.interpolate(
        img.unsqueeze(0), (new_height, new_width), mode="bilinear", antialias=True
    ).squeeze(0)


def get_free_space_gb():
    """Get free space on this disk in GB."""
    return shutil.disk_usage("/").free / 1024. / 1024. / 1024.

def clean_checkpoints(output_dir, checkpoints_total_limit, logger=None):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    checkpoints = os.listdir(output_dir)
    checkpoints = [
        d for d in checkpoints if d.startswith("checkpoint")
    ]
    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("-")[1])
    )

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = (
            len(checkpoints) - checkpoints_total_limit + 1
        )
        removing_checkpoints = checkpoints[0:num_to_remove]

        if logger is not None:
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(
                f"removing checkpoints: {', '.join(removing_checkpoints)}"
            )

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(
                output_dir, removing_checkpoint
            )
            shutil.rmtree(removing_checkpoint)

def mse_pil(pil_images1, pil_images2):
    """Compute mean squared error between two lists of PIL Image"""
    images1 = np.stack([np.asarray(elem) for elem in pil_images1]) / 256.
    images2 = np.stack([np.asarray(elem) for elem in pil_images2]) / 256.
    return np.mean((images1-images2)**2)

def elemwise_mean_pil(pil_images1, pil_images2):
    """Compute mean squared error between two lists of PIL Image"""
    images1 = np.stack([np.asarray(elem) for elem in pil_images1]) / 256.
    images2 = np.stack([np.asarray(elem) for elem in pil_images2]) / 256.
    return np.mean(images1*images2)


########################## 
# Helpers for statistics #
##########################

from tabulate import tabulate

def summarise_pipeline(pipeline):
    # Record statistics for each pipeline component which is a
    # Torch module and pretty-print them.
    pipeline_stats = []
    for comp_name, comp_val in pipeline.components.items():
        if hasattr(comp_val, 'parameters'):
            pipeline_stats.append({
                "name": comp_name,
                "dtypes": get_dtypes_from_parameters(comp_val),
                "devices": get_devices_from_parameters(comp_val),
                "# params": count_parameters(comp_val)[1],
                "# params L": count_parameters(comp_val)[0]
            })
    print(tabulate(pipeline_stats, headers="keys"))

def count_parameters_state_dict(state_dict):
    """Count number of both learnable and total parameters for a module"""
    num_params = sum([np.prod(p.size()) for p in state_dict.values()])
    return num_params

def count_parameters(model):
    """Count number of both learnable and total parameters for a module"""
    learnable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_learned_params = sum([np.prod(p.size()) for p in learnable_parameters])
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    return num_learned_params, num_params

def get_dtypes_from_parameters(module):
    """Tell me all the possible data types used by this module's parameters."""
    types = []
    for param in module.parameters():
        types.append(param.dtype)
    return set(types)

def get_devices_from_parameters(module):
    """Tell me all the possible devices that this module sits on."""
    types = []
    for param in module.parameters():
        types.append(param.device)
    return set(types)

def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024. / 1024. / 1024.
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory() / 1024. / 1024. / 1024.
    return np.nan

####################
# Model validation #
####################

import torch
import gc
import json

def load_validation_data(folderpath: str) -> Tuple[List, List, List]:
    """The metadata file is a .jsonl where each line has a 'file' and 
    'prompt' keyval pair."""
    metadata_file = os.path.join(folderpath, "metadata.jsonl")
    if not os.path.exists(metadata_file):
        raise ValueError("Cannot find metadata.jsonl in {}".format(folderpath))
    data = []
    with open(metadata_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            this_dict = json.loads(line.rstrip())
            this_dict['img'] = Image.open( os.path.join(folderpath, this_dict['file']) )
            this_dict['filename'] = os.path.join(folderpath, this_dict['file'])
            data.append(this_dict)
    return data

def log_validation(
    pipeline,
    validation_data: List[Dict],
    validation_resolution: int,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    controlnet_conditioning_scale: float = 1.0,
    scheduler_name: str = None, 
    batch_size: int = 1, 
    outfile: str = None,
    seed: int = None,
    logger=None,
    **pipe_kwargs
):
    if logger is not None:
        logger.info("Running validation... ")
        logger.info(str(pipe_kwargs))

    #pipeline.set_progress_bar_config(disable=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Pipelines can mutate scheduler state so backup the config
    # of the original scheduler and restore it after.
    original_scheduler = pipeline.scheduler.from_config(
        pipeline.scheduler.config
    )

    if scheduler_name is not None:
        # TODO: it would be much better if I can just specify the noise class
        # and instantiate it, but I'm having issues getting it to work with
        # importlib.import_module.
        print("Using scheduler: {}".format(scheduler_name))
        scheduler_class = getattr(diffusers, scheduler_name)
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

    image_logs = []

    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").\
        to(pipeline.device)

    for b, valid_example in enumerate(validation_data):

        validation_prompt = valid_example['prompt']
        validation_image = valid_example['img']
        sketchify = valid_example.get('sketchify', False)
        
        validation_image = validation_image.\
            resize((validation_resolution, validation_resolution))
        validation_image = make_alpha_all_ones(validation_image) # will be RGBA

        if sketchify:
            cond_image = ImageOps.invert(
                lineart(validation_image.convert("RGB")).resize(validation_image.size)
            )
        else:
            cond_image = validation_image.convert("RGB")

        logger.info("processing: cond_image.mode={} cond_image.size={}, sketchify = {}".format(
            cond_image.mode, cond_image.size, sketchify
        ))

        t0 = time()
        images = pipeline(
            prompt=validation_prompt,
            #image=validation_image,
            control_image=cond_image,
            generator=generator,
            guidance_scale=guidance_scale,                            # for flux-dev
            num_inference_steps=num_inference_steps,                  # for flux-dev
            width=validation_resolution,
            height=validation_resolution,
            num_images_per_prompt=batch_size,
            # https://github.com/XLabs-AI/x-flux/blob/9d9a348afa7ef3d4e09a0a40c67a349af9e28a88/src/flux/sampling.py#L147
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        logger.info(f"sanity check: images[0] has size: {images[0].size}")
        logger.info(f"time taken for generation: {time()-t0} sec")

        image_logs.append(
            {
                "validation_image": validation_image,
                "validation_prompt": validation_prompt,
                "images": images,
                "cond_image": cond_image,
                #"images_preprocessed": images_preprocessed
            }
        )

    # Save the samples to disk as well
    if outfile is not None:
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        torch.save(image_logs, outfile)

    gc.collect()
    torch.cuda.empty_cache()

    # HACK: see my comment above. Restore the original timesteps.
    pipeline.scheduler = original_scheduler

    return image_logs

########
# fsdp #
########

import os

class AccelerateIgnoreOptimizer:
    def __init__(self, accelerator, verbose=False):
        self.accelerator = accelerator
        self.verbose = verbose

    def __enter__(self):
        self.optimizer_state_backup = self.accelerator._optimizers
        if self.verbose:
            print("optimizer backup: {}".format(self.optimizer_state_backup))
        self.accelerator._optimizers = []
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator._optimizers = self.optimizer_state_backup

###################
# context manager #
###################

def get_device_memory(device_idx):
    """Return the current GPU memory occupied by tensors in GiB for a given device."""
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(device_idx) / 1024.0 / 1024.0 / 1024.0

class LogGpuMemoryAllocated:
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.t0 = time()
        self.mem_before = get_device_memory(0)

    def __exit__(self, *args):
        t1 = time()
        mem_after = get_device_memory(0)
        self.logger.info(f"{self.name} -- time taken: {t1-self.t0} sec")
        self.logger.info(f"{self.name} -- mem gain: {mem_after-self.mem_before} GiB")