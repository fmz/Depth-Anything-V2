import argparse
import sys
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

import time

from depth_anything_v2.dpt import DepthAnythingV2

def load_local_checkpoint(
        model: torch.nn.Module,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-width', type=int, default=630)
    parser.add_argument('--input-height', type=int, default=476)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    #DEVICE = 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    if os.path.isfile(args.load_from):
        load_local_checkpoint(depth_anything, args.load_from, device=DEVICE, strict=False)
        out_model_prefix = "danything_mono_metric_finetune"
    else:
        print(f"model_path={args.load_from} not found, using default timm-based weights.")
        sys.exit(1)

    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        t_start = time.time()
        depth = depth_anything.infer_image(raw_image, args.input_width, args.input_height)
        print(f'Inference time: {time.time() - t_start:.2f}s')
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)
