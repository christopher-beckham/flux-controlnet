import argparse
import os
import json
from huggingface_hub import HfApi, ModelCard

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload existing model to HuggingFace."
    )
    parser.add_argument(
        "--pretrained_cnet_checkpoint", type=str,
        required=True,
        help="Path to the checkpoint, e.g. <savedir>/<experiment>/<checkpoint>"
    )
    parser.add_argument(
        "--model_name", type=str,
        required=True,
        help="Full model id, e.g. <username>/<modelname>"
    )
    parser.add_argument(
        "--model_desc", type=str,
        default="",
        help="A textual description of the model."
    )
    parser.add_argument(
        "--model_type", type=str,
        #choices=['controlnet', 't2iadapter'],
        required=True,
        help="What model type are we trying to upload?"
    )
    
    args = parser.parse_args()
    return args

from huggingface_hub import create_repo

MODEL_CARD_TEMPLATE = """Description: {}

```
{}
```"""
 
def run(args):

    print("Checkpoint: {}".format(args.pretrained_cnet_checkpoint))
    print("Model name: {}".format(args.model_name))

    create_repo(args.model_name, repo_type="model", private=True, exist_ok=True)

    exp_cfg_path = os.path.join(
        os.path.dirname(args.pretrained_cnet_checkpoint), 
        "exp_config.json"
    )
    exp_cfg = json.loads(open(exp_cfg_path, "r").read())

    card = ModelCard(
        MODEL_CARD_TEMPLATE.format(args.model_desc, json.dumps(exp_cfg, indent=2))
    )
    card.push_to_hub(args.model_name)

    model_types = [args.model_type]
    for model_type in model_types:
        print("processing: {}".format(model_type))
        if not os.path.exists(
            os.path.join(args.pretrained_cnet_checkpoint, model_type)
        ):
            print("Could not find folder for {}, continuing...".format(
                model_type
            ))
            continue
        api = HfApi()
        api.upload_folder(
            folder_path="{}/{}".format(
                args.pretrained_cnet_checkpoint,
                model_type
            ),
            repo_id=args.model_name,
            repo_type="model",
            path_in_repo=model_type if model_type=="image_proj_model" else None
        )

if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)