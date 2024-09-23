#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Christopher Beckham. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Original source: https://github.com/linoytsaban/diffusers/blob/flux-fine-tuning/examples/dreambooth/train_dreambooth_flux.py
# (commit hash e4746830c87fa862c4892cc0b8c684646bd2f979)

import torch
import argparse
import subprocess
import diffusers
import copy
import gc
import math
import os
import json
import transformers
import accelerate
from safetensors.torch import save_file

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import numpy as np

import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from tqdm.auto import tqdm
from transformers import (
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    CLIPTextModel,
)

from time import time

from optimum.quanto import freeze, quantize, qfloat8, qint8, qint4, qint2, QTensor

from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.controlnet_flux import FluxControlNetModel as ControlNetFlux
#from diffusers import FluxControlNetPipeline

from src.pipelines.pipeline_flux_controlnet import FluxControlNetPipeline

from src.util_pil import samples_to_rows
from src.util import (
    summarise_pipeline,
    load_validation_data,
    log_validation,
    LogGpuMemoryAllocated,
    get_free_space_gb,
    count_parameters,
    count_parameters_state_dict,
    check_nan_weights,
)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxTransformer2DModel,
    # FluxPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version
)
from diffusers.utils.torch_utils import is_compiled_module

from src.util import get_device_memory

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

from src.colored_logger import get_logger
logger = get_logger(__name__)

# If we have less than this amount of free space in GB, quit script
MAX_FREE_SPACE_CUTOFF = 15.0


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def distributed_get_state_dict(accelerator, model, unwrap=True):
    if unwrap:
        model = unwrap_model(accelerator, model)
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        state_dict = model.state_dict()
    return state_dict


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_controlnet_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_controlnet_safetensors_name", type=str, default=None
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_py_file", type=str, default="dataset.py")
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument("--validation_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_single_layers", type=int, default=4)
    parser.add_argument("--control_mode", type=int, default=None, 
                        help="For a Union-ControlNet, what control mode do we assume?") 
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1 )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    # Added by Chris
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Compute and log mean/std over metrics every this many gradient steps",
    )
    parser.add_argument("--gpu_logging_steps", type=int, default=100)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["sketch"],
        default="sketch",
        help="What task are we training on?"
    )
    parser.add_argument("--save_samples_first_n_steps", type=int, default=2)
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        # default="logit_normal",
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    # parser.add_argument(
    #    "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for #text_encoder"
    # )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize everything except the controlnet?",
    )

    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default="./validation_images",
        help="Path to the validation set images / metadata file",
    )
    parser.add_argument(
        "--validation_resolution", type=int, default=1024, help="Generation resolution"
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=2,
        help="Batch size per validation image.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt, max_sequence_length=512):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
        device=device, dtype=dtype
    )
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


from datasets import load_dataset


# Copied from flat_colors_online.py
def collate_fn(examples):

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    cond_pixel_values = torch.stack(
        [example["conditioning_pixel_values"] for example in examples]
    )
    cond_pixel_values = (
        cond_pixel_values[:, 0:3].to(memory_format=torch.contiguous_format).float()
    )

    img_params = torch.stack(
        [torch.FloatTensor(example["img_params"]) for example in examples]
    )
    crop_params = torch.stack(
        [torch.FloatTensor(example["crop_params"]) for example in examples]
    )

    prompts = [example["text"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": cond_pixel_values,
        "prompts": prompts,
        "img_params": img_params,
        "crop_params": crop_params,
    }


# Copied from flat_colors_online.py
def get_dataset(
    train_data_dir: str,
    accelerator,
    dataset_py_file: str = "dataset.py",
    seed: int = None,
):
    """Load dataset in, shuffle it, and clamp if necessary."""
    dataset = load_dataset(
        path="{}/{}".format(train_data_dir, dataset_py_file),
        data_dir="{}/data".format(train_data_dir),
    )
    # NOTE(Chris): why is the main_process_first call needed??
    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=seed)
        # if args.max_train_samples is not None:
        #    train_dataset = train_dataset.select(range(max_train_samples))
    return train_dataset


from typing import Union


def load_pipeline(
    accelerator,
    pretrained_model_name_or_path: str,
    revision: str,
    variant: str,
    pretrained_controlnet_name_or_path: Union[str, None] = None,
    controlnet_checkpoint: str = None,
):

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )

    text_encoder_two = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        revision=revision,
        variant=variant,
    )

    # NOTE(Chris): if we use FSDP and have > 1 gpu, we have to do this. This is currently
    # an unsolved bug: https://github.com/huggingface/transformers/issues/33376
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        rank = torch.cuda.current_device()
        model_state_list = [
            text_encoder_one.state_dict(),
            text_encoder_two.state_dict(),
        ]
        logger.warning("HACK: broadcasting te1 and te2 to other ranks...")
        accelerate.utils.broadcast_object_list(model_state_list, 0)
        # After broadcasting, assign the state_dict in all ranks
        if rank != 0:
            logger.info(f"{rank}: Received model state_dict")
            text_encoder_one.load_state_dict(model_state_list[0])
            text_encoder_two.load_state_dict(model_state_list[1])
        check_nan_weights(text_encoder_one, "text_encoder_one")
        check_nan_weights(text_encoder_two, "text_encoder_two")
        check_nan_weights(vae, "vae")
        check_nan_weights(transformer, "transformer")

    # controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
    if pretrained_controlnet_name_or_path is not None:
        logger.info(
            f"loading pretrained controlnet from: {pretrained_controlnet_name_or_path}"
        )
        controlnet = ControlNetFlux.from_pretrained(pretrained_controlnet_name_or_path)
    else:
        params = dict(
            in_channels=64,
            pooled_projection_dim=768,
            joint_attention_dim=4096,
            # hidden_size=3072, # no exist
            # mlp_ratio=4.0,
            attention_head_dim=128,
            num_attention_heads=24,
            num_layers=args.num_layers,  # 19
            num_single_layers=args.num_single_layers,  # 38
            axes_dims_rope=[16, 56, 56],
            # theta=10_000,    # no exist
            # qkv_bias=True,   # no exist
            guidance_embeds=True,
        )

        # logger.info(f"new controlnet from scratch: {params}")
        # controlnet = ControlNetFlux(**params)
        controlnet = ControlNetFlux(**params)

    if controlnet_checkpoint is not None:
        logger.info("controlnet_ckpt is: {}".format(controlnet_checkpoint))
        ckpt_file = hf_hub_download(
            controlnet_checkpoint, filename="controlnet.safetensors"
        )
        state_dict = load_file(ckpt_file)
        controlnet.load_state_dict(state_dict)

    pipeline = FluxControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        controlnet=controlnet,
    )
    pipeline.set_progress_bar_config(disable=False)

    return pipeline


def main(args):
    disable_neptune = True
    if "NEPTUNE_PROJECT" in os.environ:
        try:
            import neptune
            from neptune.types import File
        except ImportError:
            logger.warning("Tried to import neptune but failed. If you require this, please install " + \
                "via `pip install neptune`")
        disable_neptune = False

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        # project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "exp_config.json"), "w") as f:
                f.write(json.dumps(vars(args)))

    pipeline = load_pipeline(
        accelerator,
        args.pretrained_model_name_or_path,
        args.revision,
        args.variant,
        args.pretrained_controlnet_name_or_path,
        args.pretrained_controlnet_safetensors_name,
    )

    transformer = pipeline.components["transformer"]
    vae = pipeline.components["vae"]
    text_encoder_one = pipeline.components["text_encoder"]
    text_encoder_two = pipeline.components["text_encoder_2"]
    tokenizer_one = pipeline.components["tokenizer"]
    tokenizer_two = pipeline.components["tokenizer_2"]
    controlnet = pipeline.components["controlnet"]

    if hasattr(controlnet, 'union') and controlnet.union:
        if args.control_mode is None:
            raise ValueError(
                "controlnet appears to be union variant, so args.control_mode needs to be specified"
            )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    transformer = transformer.to(dtype=weight_dtype)
    vae = vae.to(dtype=weight_dtype)
    text_encoder_one = text_encoder_one.to(dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(dtype=weight_dtype)
    controlnet = controlnet.to(dtype=weight_dtype)

    # We explicitly do the quantisation step on the CPU so we don't get OOM.
    if args.quantize:
        # https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md
        # "Alternatively, you can go ham on quantisation here and run them [text encoders] in
        # int4 or int8 mode, because no one can stop you.""
        logger.info("quantize all base models...")
        quantize(transformer, weights=qint8)
        quantize(text_encoder_one, weights=qint8)
        quantize(text_encoder_two, weights=qint8)
        quantize(vae, weights=qint8)
        freeze(transformer)
        freeze(text_encoder_one)
        freeze(text_encoder_two)
        freeze(vae)

    with LogGpuMemoryAllocated("to device", logger):

        pipeline = pipeline.to(accelerator.device)  # dtype=weight_dtype)
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        controlnet.requires_grad_(True)

    logger.info(f"controlnet type before prepare(): {type(controlnet)}")
    logger.info(
        "before prepare, controlnet has apparently {} params".format(
            count_parameters(controlnet)[-1]
        )
    )

    #pipeline.controlnet = controlnet

    if args.gradient_checkpointing:
        # NOTE(Chris): this has not been tested with any of the other flags.
        logger.info("Enabling gradient checkpointing with base transformer...")
        transformer.enable_gradient_checkpointing()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            if isinstance(unwrap_model(model), ControlNetFlux):
                load_model = ControlNetFlux.from_pretrained(
                    input_dir, subfolder="controlnet"
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")
            del load_model

            gc.collect()
            torch.cuda.empty_cache()

    # accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Optimization parameters
    params_to_optimize = controlnet.parameters()

    if args.optimizer == "adamw":
        from torch import optim

        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8 bit ADAM...")
        else:
            optimizer_class = torch.optim.AdamW
    else:
        raise NotImplementedError()

    with LogGpuMemoryAllocated("opt", logger):
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            lr=args.learning_rate,
        )

    train_dataset = get_dataset(
        train_data_dir=args.dataset_dir,
        accelerator=accelerator,
        dataset_py_file=args.dataset_py_file,
        seed=args.seed,
        # max_train_samples=args.max_train_samples
    )

    if args.task == "unified":
        # from src.tasks.flat_color import add_transform_to_dataset
        raise NotImplementedError("Not supported in this release")
    elif args.task == "sketch":
        from src.tasks.sketch import add_transform_to_dataset
    else:
        raise NotImplementedError("Unknown task: {}".format(args.task))

    # getattr(class, prepare_train_dataset)
    train_dataset = add_transform_to_dataset(
        train_dataset, resolution=args.resolution, accelerator=accelerator
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        # pin_memory=True,
        num_workers=args.num_workers,
        # prefetch_factor=8
    )

    # if not args.train_text_encoder:
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    exp_name = "/".join(args.output_dir.split("/")[-2:])
    print("exp_name: {}".format(exp_name))
    exp_group = exp_name.split("/")[0]
    exp_id = exp_name.split("/")[1]

    #########################
    # Setup Neptune logging #
    #########################

    if accelerator.is_main_process:
        if not disable_neptune:
            run = neptune.init_run(
                # project="beckhamc/test100",
                project=os.environ["NEPTUNE_PROJECT"],
                # custom_id?
                # custom_run_id=exp_name.replace("/", "--"),
                custom_run_id=exp_id,
                # exp_group--exp_id--eval--time()
                tags=[
                    "exp={}".format(exp_name),
                    "exp_id={}".format(exp_id),
                    # This means we log _during_ training.
                    "mode=train",
                ],
            )  # your credentials

            # Store experiment names.
            run["exp/name"] = exp_name
            run["exp/group"] = exp_group
            run["exp/id"] = exp_id

            # Store the argparse args here.
            for k, v in vars(args).items():
                run["argparse/{}".format(k)] = v

            # Store git branch
            run["git/branch"] = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                .decode()
                .rstrip()
            )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # We need to do this much earlier than usual in the script because we don't
            # load in the controlnet per shard.

            logger.info(
                f"Resuming from checkpoint {os.path.join(args.output_dir, path)}"
            )

            # accelerator.load_state(os.path.join(args.output_dir, path))

            load_model = ControlNetFlux.from_pretrained(
                os.path.join(args.output_dir, path), subfolder="controlnet-cuda:0"
            )
            controlnet.register_to_config(**load_model.config)
            controlnet.load_state_dict(load_model.state_dict())

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    controlnet = accelerator.prepare(controlnet)

    logger.info(f"controlnet type after prepare(): {type(controlnet)}")
    logger.info(
        "after prepare, controlnet has apparently {} params".format(
            count_parameters(controlnet)[-1]
        )
    )
    logger.info(
        "after prepare, transformer has apparently {} params".format(
            count_parameters(transformer)[-1]
        )
    )
    logger.info(
        "after prepare, unwrap_model(controlnet) has apparently {} params".format(
            count_parameters(unwrap_model(accelerator, controlnet))[-1]
        )
    )

    # In order for us to use log_validation (which in turn calls pipeline()), we need
    # to ensure the "FSDP-wrapped" version of the controlnet is used instead. So just
    # assign pipeline.controlnet to this new one. Note that this has no effect under
    # non-FSDP since we're just doing self-assignment.
    pipeline.controlnet = controlnet
    pipeline.transformer = transformer

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    validation_data = load_validation_data(args.validation_path)

    if accelerator.is_main_process:
        f_log = open(os.path.join(args.output_dir, "results.jsonl"), "a")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    verbose = False
    if "DEBUG" in os.environ and int(os.environ["DEBUG"]) == 1:
        verbose = True

    summarise_pipeline(pipeline)
    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        metric_accum = dict(loss=[], lr=[])
        t0 = time()

        for step, batch in enumerate(train_dataloader):

            if accelerator.is_main_process and step < args.save_samples_first_n_steps:
                logger.info(f"step=={step}, dumping batch to disk...")
                torch.save(batch, os.path.join(args.output_dir, f"samples-{step}.pt"))

            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                cond_pixel_values = batch["conditioning_pixel_values"].to(
                    dtype=vae.dtype
                )
                prompts = batch["prompts"]

                if accelerator.is_main_process and step == 0:
                    logger.info(
                        f"pixel_values:      {pixel_values.shape} {pixel_values.min()} {pixel_values.max()}"
                    )
                    logger.info(
                        f"cond_pixel_values: {cond_pixel_values.shape} {cond_pixel_values.min()} {cond_pixel_values.max()}"
                    )

                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = (
                        compute_text_embeddings(
                            prompts,
                            text_encoders,
                            tokenizers,
                            max_sequence_length=args.max_sequence_length,
                        )
                    )

                # 1. Convert the actual noised image to latent space.
                with torch.no_grad():

                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

                    latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2],
                        model_input.shape[3],
                        accelerator.device,
                        weight_dtype,
                    )
                    # 2. Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # 3. Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=model_input.device
                    )

                    # 4. Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(
                        timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    packed_noisy_model_input = FluxControlNetPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )

                # 5. Handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor(
                        [args.guidance_scale], device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                control_mode = None
                if args.control_mode is not None:
                    control_mode = torch.tensor([args.control_mode], device=accelerator.device).long()
                    control_mode = control_mode.expand(model_input.shape[0]).view(-1, 1)

                # 6. Prepare the control image here. We need to actually encode it with
                # the VAE, as this is what happens in the pipeline.
                with torch.no_grad():

                    if step == 1:
                        logger.info(
                            f"control image min max: {cond_pixel_values.min()} {cond_pixel_values.max()}"
                        )

                    control_image = vae.encode(cond_pixel_values).latent_dist.sample()
                    control_image = (
                        control_image - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    height_control_image, width_control_image = control_image.shape[2:]
                    control_image = FluxControlNetPipeline._pack_latents(
                        control_image,
                        batch_size=control_image.shape[0],
                        num_channels_latents=transformer.config.in_channels // 4,
                        height=height_control_image,
                        width=width_control_image,
                    )

                controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=packed_noisy_model_input,  # (bs, 1024, 64)
                    controlnet_cond=control_image,  # (bs, 1024, 64)
                    controlnet_mode=control_mode,
                    conditioning_scale=1.0,
                    timestep=timesteps / 1000,  # (bs,)
                    guidance=guidance,  # (bs,)
                    pooled_projections=pooled_prompt_embeds,  # (bs, 768)
                    encoder_hidden_states=prompt_embeds,  # (bs, 512, 4096)
                    # HACK: just take item 0 in batch dim (before: (bs,512,3)
                    txt_ids=text_ids[0],  # (512, 3)
                    img_ids=latent_image_ids,  # (1024, 3)
                    return_dict=False,
                )

                if verbose:
                    print(
                        f"controlnet_block_samples: {[ x.shape for x in controlnet_block_samples ]}"
                    )
                    print(
                        f"controlnet_single_block_samples: {[ x.shape for x in controlnet_single_block_samples ]}"
                    )

                controlnet_block_samples = [
                    x.to(dtype=packed_noisy_model_input.dtype)
                    for x in controlnet_block_samples
                ]
                controlnet_single_block_samples = [
                    x.to(dtype=packed_noisy_model_input.dtype)
                    for x in controlnet_single_block_samples
                ]

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    # HACK: just take item 0 in batch dim (before: (bs,512,3)
                    txt_ids=text_ids[0],
                    img_ids=latent_image_ids,
                    # joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                model_pred = FluxControlNetPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if step % args.gpu_logging_steps == 0:
                    logger.info(
                        f"gpu mem after loss    : {get_device_memory(accelerator.device)} GiB"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if step % args.gpu_logging_steps == 0:
                    logger.info(
                        f"gpu mem after backward: {get_device_memory(accelerator.device)} GiB"
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                summary_stats = {"epoch": epoch, "step": global_step}

                if global_step % args.checkpointing_steps == 0:

                    if get_free_space_gb() <= MAX_FREE_SPACE_CUTOFF:
                        raise Exception("Insufficient disk space detected, quitting...")

                    # NOTE: This is not in the accelerator.is_main_process if statement,
                    # we need all processes here in order to get the state dict for
                    # controlnet.
                    accelerator.wait_for_everyone()   # is it needed?
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    controlnet_state_dict = accelerator.get_state_dict(
                        controlnet, unwrap=False
                    )
                    # NOTE(Chris): I am extremely skeptical of FSDP at this point, we
                    # should actually see that the gathered state dict is correct.
                    logger.warning(
                        "# params of 'gathered' controlnet_state_dict: {}".format(
                            count_parameters_state_dict(controlnet_state_dict)
                        )
                    )
                    logger.warning(
                        "# keys of 'gathered' controlnet_state_dict: {}".format(
                            len(controlnet_state_dict.keys())
                        )
                    )
                    logger.info(
                        f"Saving state to {save_path} under {accelerator.device}"
                    )

                    rank = torch.cuda.current_device()
                    logger.info(f"rank: {rank}")

                    # controlnet.save_pretrained(state_dict=...() does nothing with
                    # state_dict in diffusers, it's a load of bs. See:
                    # https://github.com/huggingface/accelerate/issues/3089
                    if accelerator.is_main_process:
                        save_dir_actual = os.path.join(
                            save_path, "controlnet-{}".format(accelerator.device)
                        )
                        if not os.path.exists(save_dir_actual):
                            os.makedirs(save_dir_actual)
                        controlnet.save_config(save_dir_actual)
                        save_file(
                            controlnet_state_dict,
                            os.path.join(
                                save_dir_actual, "diffusion_pytorch_model.safetensors"
                            ),
                        )

                    logger.info("end saving state ...")

                if global_step % args.validation_steps == 0 or global_step == 100:
                    # with torch.autocast("cuda"):
                    samples = log_validation(
                        pipeline,
                        validation_data=validation_data,
                        validation_resolution=args.validation_resolution,
                        batch_size=args.validation_batch_size,
                        guidance_scale=args.guidance_scale,
                        controlnet_conditioning_scale=args.validation_conditioning_scale,
                        # outfile=args.outfile,
                        verbose=True,
                        seed=args.seed,
                        logger=logger,
                        # **json.loads(args.pipe_kwargs)
                    )
                    if accelerator.is_main_process:
                        if not disable_neptune:
                            samples = samples_to_rows(samples)
                            for b in range(len(samples)):
                                run["val/img{}".format(b)].append(
                                    File.as_image(samples[b]),
                                    step=global_step,
                                    name=validation_data[b]["filename"],
                                    description=validation_data[b]["prompt"],
                                )

                if (
                    global_step % args.logging_steps == 0
                    and accelerator.is_main_process
                ):
                    # Log metrics to jsonl file.
                    time_taken = time() - t0
                    for k, v in metric_accum.items():
                        summary_stats[k + "_mean"] = np.mean(v)
                        summary_stats[k + "_std"] = np.std(v)
                    summary_stats["time"] = time_taken
                    f_log.write(json.dumps(summary_stats) + "\n")
                    f_log.flush()
                    logger.info(summary_stats)
                    # Ok now log to Neptune
                    if not disable_neptune:
                        for k, v in summary_stats.items():
                            run["train/{}".format(k)].append(v, step=global_step)
                    # Also reset accumulator
                    for key in metric_accum:
                        metric_accum[key] = []

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                metric_accum["loss"].append(loss.detach().item())
                metric_accum["lr"].append(lr_scheduler.get_last_lr()[0])

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method

    # Needed for real time data augmentation (for data workers to have access)
    set_start_method("spawn")

    args = parse_args()
    main(args)
