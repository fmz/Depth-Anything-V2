#!/usr/bin/env python3
import argparse
import torch
import os
import sys
from depth_anything_v2.dpt import DepthAnythingV2

def load_local_checkpoint(
        checkpoint_path: str,
        device: torch.device,
        strict=False) -> None:
    """
    Loads a local checkpoint into the model's state_dict.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device for loading.
        strict (bool): Whether to enforce that all keys match exactly.
    """

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # TODO!: Add support for loading from other encoders
    encoder = 'vitl'  # Default to vitl, can be changed based on your needs
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 20})

    if not os.path.isfile(checkpoint_path):
        print(f"model_path={checkpoint_path} not found!.")
        sys.exit(1)

    print(f"Attempting to load checkpoint from {checkpoint_path} with strict={strict}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint was saved with a dictionary containing "model_state" or similar
    if "model_state" in ckpt:
        model_sd = ckpt["model_state"]
    else:
        model_sd = ckpt  # assume it's a direct state_dict

    missing, unexpected = model.load_state_dict(model_sd, strict=strict)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
    print("Checkpoint loaded.")

    model = model.to(device).eval()

    return model


def convert_and_save(model, example_input, output_path, method="script"):
    """
    Convert model to TorchScript and save.

    - method="script" uses torch.jit.script (handles control flow)
    - method="trace" uses torch.jit.trace   (records ops from example_input)
    """
    if method == "script":
        scripted = torch.jit.script(model)  # full Python subset analysis
    else:
        scripted = torch.jit.trace(model, example_input)  # faster for static graphs

    torch.jit.save(scripted, output_path)
    print(f"Saved TorchScript model ({method}) to {output_path}")

def parse_args():
    p = argparse.ArgumentParser("Convert PyTorch .pth to TorchScript .pt")
    p.add_argument("checkpoint", help="Path to input .pth checkpoint")
    p.add_argument("output",     help="Path to output TorchScript .pt")
    p.add_argument("--method",   choices=["script","trace"], default="script",
                   help="Conversion method: scripting or tracing")
    p.add_argument("--batch",    type=int, default=1,
                   help="Batch size for example input (for tracing)")
    p.add_argument("--height",   type=int, default=480,
                   help="Height of example input tensor")
    p.add_argument("--width",    type=int, default=640,
                   help="Width of example input tensor")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_local_checkpoint(args.checkpoint, device)

    # Prepare dummy input for tracing
    example = torch.rand(args.batch, 3, args.height, args.width, device=device, dtype=torch.float32)

    convert_and_save(model, example, args.output, method=args.method)

