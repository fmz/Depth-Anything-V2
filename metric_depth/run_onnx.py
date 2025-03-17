import argparse
import sys
import cv2
import glob
import matplotlib
import numpy as np
import os
import time
import onnxruntime as ort

def preprocess_image(image, input_width, input_height):
    """
    Preprocess the image as required by the model.
    This example converts from BGR (OpenCV) to RGB, resizes the image,
    converts the result to float32, scales pixel values to [0,1] and
    rearranges dimensions to NCHW.
    Adjust normalization if your model requires other preprocessing.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to the desired dimensions
    image_resized = cv2.resize(image_rgb, (input_width, input_height))
    # Convert to float32 and scale to [0,1]
    image_float = image_resized.astype(np.float32) / 255.0
    # Rearrange dimensions from HWC to CHW
    image_chw = np.transpose(image_float, (2, 0, 1))
    # Add batch dimension: (1, C, H, W)
    input_tensor = np.expand_dims(image_chw, axis=0)
    return input_tensor

def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 Metric Depth Estimation using ONNX"
    )
    parser.add_argument('--img-path', type=str, required=True,
                        help="Path to an image file or a directory (or a text file listing image paths)")
    parser.add_argument('--input-width', type=int, default=630,
                        help="Input width for the model")
    parser.add_argument('--input-height', type=int, default=476,
                        help="Input height for the model")
    parser.add_argument('--outdir', type=str, default='./vis_depth',
                        help="Directory to save visualization images")
    parser.add_argument('--onnx-model', type=str, required=True,
                        help="Path to the exported ONNX model file")
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true',
                        help='Save the raw model output as a numpy file')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true',
                        help='Save only the depth prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true',
                        help='Use grayscale visualization (instead of color)')
    args = parser.parse_args()

    # Create ONNX Runtime session; try to use GPU if available.
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx_model, providers=providers)
    # Get input and output names from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Get image filenames from a file, directory, or single file path
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
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
        if raw_image is None:
            print(f"Warning: failed to load image {filename}")
            continue

        # Preprocess the image to create a 4D input tensor
        input_tensor = preprocess_image(raw_image, args.input_width, args.input_height)

        t_start = time.time()
        # Run inference using ONNX Runtime; note the input dict keys must match the model's input names.
        onnx_outputs = session.run([output_name], {input_name: input_tensor})
        inference_time = time.time() - t_start
        print(f'Inference time: {inference_time:.2f}s')

        depth = onnx_outputs[0].squeeze()

        # Optionally save the raw depth output
        if args.save_numpy:
            np_output_path = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy'
            )
            np.save(np_output_path, depth)

        # Normalize the depth for visualization
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        depth_norm = depth_norm.astype(np.uint8)

        if args.grayscale:
            depth_vis = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
        else:
            # Apply the Spectral colormap and convert from RGB to BGR for OpenCV
            depth_vis = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(
            args.outdir,
            os.path.splitext(os.path.basename(filename))[0] + '.png'
        )
        if args.pred_only:
            cv2.imwrite(output_path, depth_vis)
        else:
            # Create a white split region and concatenate the original image and the depth map
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
            cv2.imwrite(output_path, combined_result)

if __name__ == '__main__':
    main()
