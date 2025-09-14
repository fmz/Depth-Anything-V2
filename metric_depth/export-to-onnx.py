import argparse
import sys
import torch
import os
import onnx
import onnxconverter_common 


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


def export_depthanything(
    model,
    output_path,
    batch_size=1,
    height=640,
    width=480,
    dynamic=False,
    fp16=False,
    opset=15,
    device=torch.device("cpu")
):
    model.eval()

    # if height % 14 != 0 or width % 14 != 0:
    #     raise ValueError("Input height and width must be divisible by 14")
    
    # Determine dummy input shape and dtype
    bs = batch_size if batch_size and batch_size > 0 else 1
    hh = height if height and height > 0 else 518
    ww = width if width and width > 0 else 518
    dtype = torch.float32
    dummy_input = torch.randn(bs, 3, hh, ww, dtype=dtype).to(device)
    
    # Configure dynamic axes if requested
    if dynamic or batch_size == 0 or height == 0 or width == 0:
        dynamic_axes = {
            "rgb":   {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            "depth": {0: "batch_size", 2: "height", 3: "width"}
        }
    else:
        dynamic_axes = None

    print(f"Exporting with batch_size={bs}, height={hh}, width={ww}, opset={opset}, dynamic={dynamic}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["rgb"],
        output_names=["depth"],
        dynamic_axes=dynamic_axes
    )
    print(f"Model exported to {output_path} as FP32")
        
    # If fp16 option is requested, convert the ONNX model to mixed precision:
    # TODO: Make this a separate script
    if fp16:
        # Load the exported FP32 model
        model_fp32 = onnx.load(output_path)
        # Convert internal tensors (initializers) to FP16 while keeping inputs/outputs as FP32.
        # model_mixed = float16_converter.convert_float_to_float16(
        #     model_fp32, keep_io_types=True
        # )
        output_fp16_path = output_path.replace(".onnx", "_fp16.onnx")
        # onnx.save(model_mixed, output_fp16_path)
        feed_dict = {'rgb': dummy_input.detach().cpu().numpy()}
        model_fp16 = onnxconverter_common.auto_convert_mixed_precision(model_fp32, feed_dict, rtol=0.5, atol=0.00001, keep_io_types=True)
        onnx.save(model_fp16, output_fp16_path)
        print(f"Model converted to mixed precision (FP16 weights, FP32 I/O) and saved to {output_fp16_path}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DepthAnythingV2 to ONNX")
    parser = argparse.ArgumentParser(description="Export DepthAnythingV2 to ONNX")
    parser.add_argument("--load-from", type=str, required=True,
                        help="Path to the .pth checkpoint for DepthAnythingV2.")
    parser.add_argument("--encoder", type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Type of ViT encoder in the DepthAnythingV2 model.")
    parser.add_argument("--max-depth", type=float, default=20.0,
                        help="Max depth value for DepthAnythingV2.")

    # ONNX arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy input (use 0 for dynamic batch)")
    parser.add_argument("--height", type=int, default=518, help="Input image height (use 0 for dynamic height)")
    parser.add_argument("--width", type=int, default=518, help="Input image width (use 0 for dynamic width)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch, height, and width")
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 half-precision")
    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version to use")
    parser.add_argument("--output", type=str, default="DepthAnythingV2.onnx", help="Output ONNX file path")
    args = parser.parse_args()
    # set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # prepare config from your original dictionary
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # instantiate DepthAnythingV2
    depth_anything = DepthAnythingV2(
        **{
            **model_configs[args.encoder],
            'max_depth': args.max_depth
        }
    )
    
    # load checkpoint
    if os.path.isfile(args.load_from):
        load_local_checkpoint(depth_anything, args.load_from, device=DEVICE, strict=False)
        out_model_prefix = "danything_mono_metric_finetune"
    else:
        print(f"model_path={args.load_from} not found, using default timm-based weights.")
        sys.exit(1)

    depth_anything.to(DEVICE)

    # Export
    export_depthanything(
        model=depth_anything,
        output_path=args.output,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        dynamic=args.dynamic,
        fp16=args.fp16,
        opset=args.opset,
        device=DEVICE
    )