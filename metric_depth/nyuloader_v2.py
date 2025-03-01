import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

from torchvision.transforms import v2

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

from cv2.ximgproc import guidedFilter

from utils import save_depth

num_iters = 0

def guided_filter(rgb_img, depth_img, radius=2, eps=1e-3):
    """
    Example wrapper if you have an opencv ximgproc guided filter or a custom guided filter code.

    Args:
        rgb_img (np.ndarray): The guidance image, shape (H, W, 3) or (H,W) if grayscale
        depth_img (np.ndarray): The input depth image to refine, shape (H, W)
        radius (int): Window radius
        eps (float): Regularization constant
    Returns:
        np.ndarray: The refined depth image with sharper edges, shape (H, W)
    """
    rgb_img = rgb_img.transpose((1,2,0))

    # guided = guidedFilter(
    #     guide=rgb_img,
    #     src=depth_img,
    #     radius=radius,
    #     eps=eps
    # )

    # If not, we can approximate with a joint bilateral filter:
    guided = cv2.ximgproc.jointBilateralFilter(
        joint=rgb_img.astype(np.float32),
        src=depth_img.astype(np.float32),
        d=2*radius+1,
        sigmaColor=0.2,
        sigmaSpace=radius
    )


    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 2))
    # # Erode then dilate is called "open"
    # guided = cv2.morphologyEx(depth_img, cv2.MORPH_OPEN, kernel)

    global num_iters
    if (num_iters % 50 == 0):
        num_iters = 0
        save_depth(depth_img, 'tmp/in_depth_img.png')
        save_depth(guided, 'tmp/guided_depth_img.png')
    num_iters += 1

    return guided

class ResizeAndCrop:
    """
    A custom transform that first resizes an image to (target_height, target_width)
    and then center–crops it to (crop_height, crop_width).
    """
    def __init__(
        self,
        target_height: int,
        target_width: int,
        crop_height: int,
        crop_width: int,
        resize_target: bool = False,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resize_method: str = "minimal",
        image_interpolation_method = cv2.INTER_CUBIC,
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.resize_target = resize_target
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        # First, resize to the full target dimensions.
        image = cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=self.image_interpolation_method,
        )
        # Then compute center crop coordinates to get dimensions divisible by factor.
        top = (self.target_height - self.crop_height) // 2
        left = (self.target_width - self.crop_width) // 2
        image = image[top: top + self.crop_height, left: left + self.crop_width]
        sample["image"] = image
        return sample

# Modified Dataset class integrating resize+crop into one shot.
class NYUDepthDataset(Dataset):
    """
    A dataset for NYU-based depth completion or depth estimation tasks.
    It loads:
      - RGB images
      - Lidar / sparse depth
      - GT depth (optional, if available)
      - Optional mask arrays for applying custom masks

    Key features:
      - Add random noise to sparse depth
      - Apply random or fixed masks
      - Possibly resize and normalize
    """

    def __init__(
        self,
        data_dir: str,
        mode: str,
        use_mask: bool = False,
        add_noise: bool = False,
        height: int = 480,
        width: int = 640,
        mask_dir: str = None,
        resize: bool = True,
        divisible_by: int = None,
    ):
        """
        Args:
            data_dir (str): root directory to the dataset.
            mode (str): "train", "val", or "test".
            use_mask (bool): whether to apply a binary mask to the depth.
            add_noise (bool): whether to add random multiplicative noise to depth.
            height, width (int): target image size. (480x640 by default)
            mask_dir (str): if masks are in a separate folder. If None, tries data_dir + '/mask'.
            resize (bool): whether to force a Resize transform to (width, height).
        """
        super().__init__()

        # 1) Paths.
        self.data_dir = data_dir
        self.mode = mode
        self.use_mask = use_mask
        self.add_noise = add_noise

        self.rgb_path = os.path.join(data_dir, mode, "img")
        self.lidar_path = os.path.join(data_dir, mode, "depth")  # sparse depth
        self.gt_path = os.path.join(data_dir, mode, "gt")        # ground-truth depth
        self.mask_path = mask_dir if mask_dir else os.path.join(data_dir, "mask")

        # 2) Collect file lists.
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_path, "*.npy")))
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_path, "*.png")))
        self.gt_files = sorted(glob.glob(os.path.join(self.gt_path, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_path, "*.npy")))

        if len(self.rgb_files) == 0:
            raise RuntimeError(f"No RGB images found in {self.rgb_path}")
        if len(self.lidar_files) == 0:
            raise RuntimeError(f"No Lidar depth found in {self.lidar_path}")

        # 3) Image size & crop: if divisible_by is provided, we compute the final dimensions.
        self.height = height
        self.width = width
        self.resize = resize
        self.divisible_by = divisible_by

        # 4) Transforms.
        transform_list = []
        if self.resize:
            if self.divisible_by is not None and self.divisible_by > 0:
                # Compute final crop dimensions that are divisible by the factor.
                new_H = (self.height // self.divisible_by) * self.divisible_by
                new_W = (self.width // self.divisible_by) * self.divisible_by
                # Use our combined transform.
                transform_list.append(
                    ResizeAndCrop(
                        target_height=self.height,
                        target_width=self.width,
                        crop_height=new_H,
                        crop_width=new_W,
                        resize_target=False,
                        keep_aspect_ratio=False,
                        ensure_multiple_of=1,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    )
                )
            else:
                transform_list.append(
                    Resize(
                        width=self.width,
                        height=self.height,
                        resize_target=False,
                        keep_aspect_ratio=False,
                        ensure_multiple_of=1,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    )
                )
        transform_list.append(
            #NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        transform_list.append(PrepareForNet())  # Converts to float32 and permutes to [C,H,W].
        self.rgb_transform = v2.Compose(transform_list)

    def __len__(self):
        return min(len(self.rgb_files), len(self.lidar_files))

    def __getitem__(self, index):
        # 1) Load file paths.
        rgb_path = self.rgb_files[index]
        lidar_path = self.lidar_files[index]
        gt_path = self.gt_files[index] if index < len(self.gt_files) else None

        # 2) Load and transform RGB.
        rgb, rgb_numpy = self.load_rgb(rgb_path)  # Already resized & cropped.

        # 3) Load lidar depth.
        depth_sparse = self.load_npy_depth(lidar_path)

        # 4) Optionally load ground-truth depth.
        depth_gt = None
        if gt_path and os.path.isfile(gt_path):
            depth_gt = self.load_npy_depth(gt_path, rgb_np=rgb_numpy)

        # 5) Preprocess depth (noise and/or mask).
        depth_sparse, mask = self.preprocess_depth(depth_sparse)

        # 6) Resize (and crop) depth and mask to match final RGB dimensions.
        # Since rgb is already processed, use its H and W.
        final_H, final_W = rgb.shape[1], rgb.shape[2]
        if self.resize:
            depth_sparse = self.resize_depth(depth_sparse, final_H, final_W)
            if depth_gt is not None:
                depth_gt = self.resize_depth(depth_gt, final_H, final_W)
            mask = self.resize_depth(mask, final_H, final_W)

        depth_sparse = depth_sparse.squeeze()
        mask         = mask.squeeze()

        sample = {
            "rgb": rgb,            # FloatTensor [3,H,W]
            "depth": depth_sparse, # FloatTensor [1,H,W] (sparse)
            "mask": mask,          # FloatTensor [1,H,W]
        }
        if depth_gt is not None:
            depth_gt = depth_gt.squeeze()
            sample["gt"] = depth_gt



        return sample

    # -------------------------------------------------------------
    #                HELPER FUNCTIONS
    # -------------------------------------------------------------
    def load_rgb(self, path: str) -> torch.FloatTensor:
        """
        Loads an RGB image from path, converts from BGR->RGB, [0..1],
        then applies self.rgb_transform (Resize + Normalize + PrepareForNet).
        Returns a FloatTensor with shape [3,H,W].
        """
        img = cv2.imread(path)  # BGR, shape (H,W,3)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Use dpt's transform pipeline
        sample_dict = {"image": img}  # The transform expects a dict
        sample_dict = self.rgb_transform(sample_dict)
        # "image" is now float32 [H,W,3]
        # We'll keep it as .transpose(2,0,1) done by PrepareForNet
        # So final shape is [3,H,W]
        rgb_numpy = sample_dict["image"]
        rgb_tensor = torch.from_numpy(rgb_numpy)  # already [C,H,W], float32
        return rgb_tensor, rgb_numpy

    def load_npy_depth(self, path: str, rgb_np: np.array = None) -> torch.FloatTensor:
        """
        Loads a .npy file that contains depth data of shape (H,W).
        Returns a FloatTensor [1,H,W].
        """
        arr = np.load(path)
        if arr.ndim == 2:
            pass  # shape = (H,W)
        elif arr.ndim == 3:
            # In case it's already (1,H,W) or something
            arr = arr.squeeze(0)  # just to unify

        # if rgb_np is not None:
        #     # Apply guided filter
        #     arr = guided_filter(rgb_np, arr, radius=5, eps=1e-3)

        # Expand dims
        arr = np.expand_dims(arr, axis=0)  # shape => [1,H,W]

        return torch.from_numpy(arr.astype(np.float32))

    def resize_depth(self, depth: torch.FloatTensor, new_h: int, new_w: int) -> torch.FloatTensor:
        """
        Force-resizes a depth map [1, origH, origW] to [1, newH, newW]
        using nearest-neighbor interpolation.
        """
        # Convert shape: [1,H,W] => [H,W]
        d_np = depth.squeeze(0).numpy()
        d_resized = cv2.resize(d_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(d_resized).unsqueeze(0)  # shape: [1,newH,newW]

    def get_random_mask(self) -> torch.FloatTensor:
        """
        Reads a random mask file from self.mask_files, returning a FloatTensor [1,H,W].
        If no mask files exist, returns a ones() mask (no op).
        """
        if len(self.mask_files) == 0:
            # fallback => no mask
            return torch.ones((1, 480, 640), dtype=torch.float32)
        mask_path = random.choice(self.mask_files)
        mask_arr = np.load(mask_path)  # could be shape (H,W)
        # If shape isn't 480x640, we can forcibly resize
        if mask_arr.shape != (480, 640):
            mask_img = Image.fromarray(mask_arr)
            mask_img = mask_img.resize((640, 480), Image.NEAREST)
            mask_arr = np.array(mask_img)
        # Expand dims => [1,H,W]
        mask_tensor = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(0)
        return mask_tensor

    def preprocess_depth(self, depth_sparse: torch.FloatTensor):
        """
        - If self.add_noise is True, apply random multiplicative noise to 10% of pixels.
        - If self.use_mask is True, multiply by a random mask.
          Otherwise, we do the 'partial zeroing' approach as in your old code.

        Returns: (depth, mask) => both FloatTensors [1,H,W].
          mask = 1 for valid, 0 for invalid.
        """
        # 1) Possibly add noise
        depth = depth_sparse.clone()  # avoid in-place
        mask = torch.ones_like(depth)  # default mask = all valid

        if self.add_noise:
            num_elements = depth.numel()
            num_noisy_points = int(num_elements * 0.1)
            indices = torch.randperm(num_elements)[:num_noisy_points]

            # noise range -0.1..0.1
            noise = torch.empty(num_noisy_points, dtype=torch.float32).uniform_(-0.1, 0.1)
            flattened = depth.reshape(-1)
            # multiplicative: depth[i] += depth[i]*noise
            flattened[indices] += flattened[indices] * noise

        # 2) Mask usage
        if self.use_mask:
            # multiply by a random mask from disk
            random_mask = self.get_random_mask()  # shape [1,H,W]
            # ensure it's the same shape as 'depth'
            # if shape mismatch, you might want to resize, but let's assume 480x640
            # => We'll unify after we do final resizing in __getitem__
            depth = depth * random_mask
            mask = random_mask
        else:
            # we do partial zeroing as in your old code
            # pick random points in depth to set to 0
            # the old code picks as many points as the number of zeros in mask.
            # We'll do a simpler approach:
            random_mask = self.get_random_mask()
            num_zeros_in_mask = (random_mask == 0).sum().item()
            num_elements = depth.numel()
            num_points_to_zero = min(num_zeros_in_mask, num_elements)
            indices = torch.randperm(num_elements)[:num_points_to_zero]

            flat_d = depth.reshape(-1)
            flat_d[indices] = 0.0

        return depth, mask
