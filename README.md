# flux-controlnet

Code to train a FLUX ControlNet on a 40GB A100 GPU. This codebase is a heavily modified version of Linoy Tsaban's dreambooth LoRA script.

**Disclaimer:** I make no guarantees as to the correctness or reliability of this code. As of time of writing (23/09) there is no official FLUX training code to cross-reference against, and training with weight quantisation may have non-trivial effects on sample quality. No "real" dataset is provided with this script (though an example dataset is provided), and FLUX controlnets are extremely high capacity models (ranging in the billions of parameters), which means they are extremely likely to overfit on even modestly-sized datasets. This code is primarily intended to be used with quantisation enabled on a single GPU, though you may also try it with FSDP on > 1 GPU.

## Setup

### Environment

Create a conda environment:

```
conda create -n <my env name> python=3.9
conda activate <my env name>
```

We need to install the following:

```
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/huggingface/accelerate
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/optimum-quanto
pip install datasets==2.20.0 tabulate==0.9.0 controlnet-aux==0.0.9
pip install sentencepiece protobuf bitsandbytes
```

Because Python package management is a pain, if you are still missing packages then you can also consult `requirements.txt`, and/or try install with `pip install -r requirements.txt`.

### Dataset

For this repo you should provide your own dataset of interest. I have provided an example dataset (based on `diffusers/dog-example`) just to demonstrate the training script functions as intended. 

```
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="beckhamc/dog-example", repo_type="dataset", local_dir="dog-example")'
```

The resulting directory structure should look like this inside the now-created `dog-example` folder:

```
.
├── data
│   ├── <filename1>.jpeg
│   ├── ...
│   ├── ...
│   ├── <filenameN>.jpeg
|   |── metadata.jsonl
├── dataset.py
```

You can use this dataset as an example for how to prepare your own dataset. Please see `dataset.py` and `metadata.jsonl` to see how things are structured.

### Configuration

Firstly, cd into `exps`, `cp env.sh.bak env.sh` and make sure the following variables are defined:
- `$DATA_DIR`: where your dataset is located, e.g. `/home/chris/datasets/dog-example`.
- `$SAVE_DIR`: where to save experiments, e.g. `/home/chris/datasets/results`.
- (optional) `NEPTUNE_API_KEY` for logging to Neptune. If you use Neptune, make sure to install it with `pip install neptune`.

Also, if you are dealing with gated models or want to upload them then ensure you are authenticated on huggingface via `huggingface-cli login`.

## Training

Please see `exps/train.sh` for an example script. The script usage is as follows:

```
cd exps
source env.sh # do this just once, very important
bash train.sh <name of experiment>
```

For instance, running with `bash train.sh my_experiment` gives the following printout before the rest of the code is executed:

```
-------------------------------
Exp group        :   my_experiment_123
Exp id           :   1727111804
Experiment name  :   my_experiment_123/1727111804
Output directory :   /home/chris/results/flux-adapters/my_experiment_123/1727111804
Data dir         :   /home/chris/datasets/dog-example
-------------------------------
```

i.e., `my_experiment_123/1727111804` is the unique identifier of the experiment which lives under `$SAVE_DIR`. Currently, the digit identifier is just Unix time, but you can modify this to support whatever experiment scheduler system you're using (e.g. Slurm). Note that if you invoke the script like so:

```
bash train.sh <experiment name>/<id>
```

then `<id>` will be used instead as the digit identifier. If you use Neptune to log experiments, then you can run `NEPTUNE_PROJECT=<neptune project name> bash train.sh <experiment name>` instead. 

It _should_ be possible to train on a single 40GB A100 GPU for a "modestly"-sized ControlNet. By default, the training script sets `args.num_layers=2` and `args.num_single_layers=4` (i.e. "depth 2-4"). For reference, the Union ControlNet-Pro by Shakker Labs (`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro`) uses depth 5-10 and probably won't run, but if you want to see how you could finetune on top of it then you can look at `exps/train-510.sh`.

Here is a breakdown of how we save on GPU memory:

- `--mixed_precision bf16`: everything gets cast into this precision, even the ControlNet.
- `--use_8bit_adam`: ADAM uses a lot of GPU memory since it has to store statistics for each parameter being trained. Here we use the 8-bit version.
- `--quantize`: quantise everything (except ControlNet) into `int8` via the `optimum-quanto` library. This is _weight only_ quantisation, so params are stored in `int8` and are de-quantised on the fly. You may be able to squeeze out even more savings with lower bits but this has not been tested.

### Debugging and hyperparameters

Some things to consider:
- To increase the effective batch size, you can increase `gradient_accumulation_steps` but this will slow down training by that same factor.
- You should not make the number of data loader workers too large. This is because each data loader has a copy of the line-art detector which adds to the overall GPU memory. If this is problematic, then you should preprocess the dataset and modify `dataset.py` to also load the conditioning images in.
- To bring down GPU usage even more, you could precompute the conditioning images and modify `metadata.jsonl` and `dataset.py` accordingly to directly load them in. Furthermore, you can also squeeze some savings out by precomputing the VAE and text encoder embeddings.

Various "quality of life" printouts are done during the execution of the script to make sure everything works as intended. Most notably this:

```
name            dtypes            devices                            # params    # params L
--------------  ----------------  ------------------------------  -----------  ------------
vae             {torch.bfloat16}  {device(type='cuda', index=0)}  8.38197e+07   0
text_encoder    {torch.bfloat16}  {device(type='cuda', index=0)}  1.2306e+08    0
text_encoder_2  {torch.bfloat16}  {device(type='cuda', index=0)}  4.76231e+09   0
transformer     {torch.bfloat16}  {device(type='cuda', index=0)}  1.19014e+10   0
controlnet      {torch.bfloat16}  {device(type='cuda', index=0)}  1.34792e+09   1.34792e+09
```

For each pipeline component we scan through all the parameters and list _all_ the dtypes we found*, and _all_ the devices they were found on. We also count the total number of params and learnable parameters, respectively. (* One exception however is that even with `--quantize` enabled we won't see the actual internal dtype of the weights (which should be `qint8`), this is due to a certain abstraction implemented in `optimum-quanto` which makes it so that quanto tensors "look like" `bf16` even though the internal representation is actually int.)

As you can see, under the default arguments the ControlNet is ~1,347,920,000 (~1.3B) params which is absolutely ginormous in absolute scale but still "small" in relative scale (the base transformer is ~11B params, so this amounts to ~10% of that size). From personal experience it seems as though this model size is "required" to get a decent performing model (short of a more long term solution in finding a more efficient architecture). For reference, the Shakker Labs CN-Union-Pro is much larger and sits at ~3.3B params. (While you can try to run this in `exps/train-510.sh`, it will probably fail.)

### Uploading to the Hub

Simply cd into `exps` and run:

```
bash upload.sh <experiment name>/checkpoint-<iteration> <org>/<model name> --model_type=controlnet
```

Note that `<experiment name>/checkpoint-<iteration>` is relative to `$SAVE_DIR`. `<org>` will be either your HuggingFace username or organisation name.

## Q&A

- *Can this work with Flux Schnell*: I don't suggest fine-tuning on top of Schnell since it's a distilled model and those kinds of models use different loss functions. You can train the ControlNet on top of dev and then run the ControlNet at inference time with the Schnell model loaded.
- *Why doesn't the code save the accelerator state*: because it complains about the 8-bit ADAM optimiser state. Also saving the state uses way too much disk space and it's not worth it.
- *Does this work with FSDP?* It should but I haven't tested it with quantisation with `optimum-quanto`. In my experience, training with FSDP can be quite a flaky and painful process.