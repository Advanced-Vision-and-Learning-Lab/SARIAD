from anomalib.data import Folder
from SARIAD.config import DATASETS_PATH, DEBUG
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.utils.img_utils import img_debug
from SARIAD.pre_processing.SARCNN import *

import os, cv2, random, shutil
import numpy as np
from tqdm import tqdm

NAME = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"

class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", split="train"):
        self.split = split
        self.train_batch_size = 1 if DEBUG else 32
        self.eval_batch_size = 1 if DEBUG else 16
        self.image_size = (512,512)

        fetch_blob(NAME, drive_file_id=DRIVE_FILE_ID, ext="rar")
        self.split_masks()
        self.generate_norm()

        super().__init__(
            name = NAME,
            root = f"{DATASETS_PATH}/{NAME}/{sub_dataset}",
            mask_dir = f"voc_style/JPEGImages_PSeg_GT_Mask_{self.split}",
            normal_dir = f"voc_style/JPEGImages_{self.split}_norm",
            abnormal_dir = f"voc_style/JPEGImages_{self.split}{'_' + sub_category if sub_category != '' else ''}",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
        )

        self.setup()

    def split_masks(self):
        """
        Splits the masks from the main JPEGImages_PSeg_GT_Mask directory into
        JPEGImages_PSeg_GT_Mask_train and JPEGImages_PSeg_GT_Mask_test based on
        the image names present in JPEGImages_train and JPEGImages_test.
        """
        base_root_dir = f"{DATASETS_PATH}/{NAME}/PSeg_SSDD/voc_style"
        
        source_mask_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask")
        train_masks_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask_train")
        test_masks_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask_test")

        original_train_images_dir = os.path.join(base_root_dir, "JPEGImages_train")
        original_test_images_dir = os.path.join(base_root_dir, "JPEGImages_test")

        # Check if split mask directories already exist and are populated
        if (os.path.exists(train_masks_dir) and os.listdir(train_masks_dir) and
            os.path.exists(test_masks_dir) and os.listdir(test_masks_dir)):
            print("Masks are already split. Skipping mask splitting.")
            return

        print("Splitting masks into train and test directories.")

        os.makedirs(train_masks_dir, exist_ok=True)
        os.makedirs(test_masks_dir, exist_ok=True)

        # Get the list of image files
        train_image_files = {f for f in os.listdir(original_train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
        test_image_files = {f for f in os.listdir(original_test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}

        # Iterate through all masks in the source mask directory
        mask_files = [f for f in os.listdir(source_mask_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for mask_file in tqdm(mask_files, desc="Splitting masks"):
            source_mask_path = os.path.join(source_mask_dir, mask_file)
            
            if mask_file in train_image_files:
                destination_mask_path = os.path.join(train_masks_dir, mask_file)
            elif mask_file in test_image_files:
                destination_mask_path = os.path.join(test_masks_dir, mask_file)
            else:
                print(f"Warning: Mask {mask_file} does not correspond to any image in train or test sets. Skipping.")
                continue
            shutil.copy2(source_mask_path, destination_mask_path)
        
        print("Mask splitting complete.")

    def apply_mask(self, image, mask, min_crop_size=3, max_crop_size=15, sample_step=10):
        """
        Fills in the masked areas of the image using random crops from the background,
        with random rotation, to mimic local texture.

        Parameters:
            image (np.array): The input grayscale image (H, W).
            mask (np.array): The binary mask indicating anomalous regions (1 for anomaly, 0 for background) (H, W).
                             This mask should be a copy if you intend to modify it internally.
            min_crop_size (int): Minimum side length for random background crops.
            max_crop_size (int): Maximum side length for random background crops.
            sample_step (int): Step size for sampling background patches (e.g., 1 for every pixel, larger for sparse).

        Returns:
            np.array: The image with anomalies filled in using background patches.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_float = image.astype(np.float32)

        mask_int = mask.astype(np.uint8)
        indices_not_in_mask = np.where(mask_int == 0)
        img_mean = image_float[indices_not_in_mask].mean()

        comp = lambda x: (x >= img_mean) if (img_mean/255.0 >= 0.5) else (x <= img_mean)

        h, w = mask.shape
        working_mask = mask.copy() 

        # 1. Collect Valid Background Patches
        patch_pool = []
        
        # Iterate through a grid to find valid background patches
        for y_top in range(0, h - min_crop_size + 1, sample_step):
            for x_left in range(0, w - min_crop_size + 1, sample_step):
                # Check for max_crop_size validity first
                if y_top + max_crop_size <= h and x_left + max_crop_size <= w:
                    # Check if the largest possible crop from this top-left corner is fully background
                    if np.all(mask[y_top : y_top + max_crop_size, x_left : x_left + max_crop_size] == 0):
                        # If the max_crop_size patch is background, then all smaller patches from this corner are too.
                        for k_size in range(min_crop_size, max_crop_size + 1):
                            current_patch = image_float[y_top : y_top + k_size, x_left : x_left + k_size]
                            if current_patch.shape[0] == k_size and current_patch.shape[1] == k_size: # Ensure actual size
                                if comp(current_patch.mean()):
                                    patch_pool.append(current_patch)
                else: # Near boundaries, try smaller patches that fit
                    for k_size in range(min_crop_size, max_crop_size + 1):
                        if y_top + k_size <= h and x_left + k_size <= w:
                            current_patch = image_float[y_top : y_top + k_size, x_left : x_left + k_size]
                            if np.all(mask[y_top : y_top + k_size, x_left : x_left + k_size] == 0) and \
                               current_patch.shape[0] == k_size and current_patch.shape[1] == k_size:
                                if comp(current_patch.mean()):
                                    patch_pool.append(current_patch)

        if not patch_pool:
            print("Warning: No valid background patches found for filling. Returning original image.")
            return image

        # Get all masked pixels from the original mask
        masked_y_coords, masked_x_coords = np.where(mask == 1) 
        
        # Shuffle masked pixels to randomize the filling order
        masked_pixels = list(zip(masked_y_coords, masked_x_coords))
        random.shuffle(masked_pixels)

        # Temporary canvases to accumulate pixel values and counts for averaging overlaps
        temp_fill_canvas = np.zeros_like(image_float)
        temp_weight_canvas = np.zeros_like(image_float, dtype=np.int32)

        # Iterate through shuffled masked pixels and paste patches
        for cy, cx in masked_pixels:
            # skip if already covered by a previous patch fill
            if working_mask[cy, cx] == 0: 
                 continue

            patch_to_paste = random.choice(patch_pool)
            ph, pw = patch_to_paste.shape

            # Calculate target region for pasting around the current masked pixel (cy, cx)
            target_y_start = cy - ph // 2
            target_x_start = cx - pw // 2

            # Ensure the paste region is within bounds
            ty_start = max(0, target_y_start)
            tx_start = max(0, target_x_start)
            ty_end = min(h, target_y_start + ph)
            tx_end = min(w, target_x_start + pw)

            # Adjust the slice of the patch if it's pasted near image boundaries
            patch_slice_y_start = 0 if target_y_start >= 0 else -target_y_start
            patch_slice_x_start = 0 if target_x_start >= 0 else -target_x_start
            
            current_patch_portion = patch_to_paste[
                patch_slice_y_start : patch_slice_y_start + (ty_end - ty_start),
                patch_slice_x_start : patch_slice_x_start + (tx_end - tx_start)
            ]

            # Get the actual masked region within the target paste area from the working_mask
            mask_region_for_paste = working_mask[ty_start:ty_end, tx_start:tx_end]

            if mask_region_for_paste.size > 0 and np.any(mask_region_for_paste == 1):
                # Get relative indices within the target_region where working_mask is 1
                indices_to_fill_y, indices_to_fill_x = np.where(mask_region_for_paste == 1)
                # Accumulate pixel values and weights in temporary canvases
                temp_fill_canvas[ty_start:ty_end, tx_start:tx_end][indices_to_fill_y, indices_to_fill_x] += \
                    current_patch_portion[indices_to_fill_y, indices_to_fill_x]
                temp_weight_canvas[ty_start:ty_end, tx_start:tx_end][indices_to_fill_y, indices_to_fill_x] += 1
                # Mark as filled in the working_mask to prevent re-processing
                working_mask[ty_start:ty_end, tx_start:tx_end][indices_to_fill_y, indices_to_fill_x] = 0

        # Final composition
        temp_weight_canvas[temp_weight_canvas == 0] = 1 # Avoid division by zero
        filled_region_avg = temp_fill_canvas / temp_weight_canvas

        # Combine the original image (background) with the filled masked area
        final_output_image = image_float.copy()
        final_output_image[mask == 1] = filled_region_avg[mask == 1]

        # Ensure values are within the valid image range (0-255) and convert back to original dtype
        return np.clip(final_output_image, 0, 255).astype(image.dtype)

    def generate_norm(self):
        """
        Generates synthetic normal images by filling in anomalies from original
        images using their corresponding masks for both train and test sets.
        """
        base_root_dir = f"{DATASETS_PATH}/{NAME}/PSeg_SSDD/voc_style"
        
        sets = {"train": "JPEGImages_train", "test": "JPEGImages_test"}

        # Check if both normal image directories already exist
        all_norm_dirs_exist = True
        for set_name, _ in sets.items():
            normal_images_dir = os.path.join(base_root_dir, f"JPEGImages_{set_name}_norm")
            if not os.path.exists(normal_images_dir) or not os.listdir(normal_images_dir):
                all_norm_dirs_exist = False
                break
        
        if all_norm_dirs_exist:
            print("Normal image directories already exist and contain files. Skipping generation.")
            return

        print("Could not find normal image directories, generating.")

        for set_name, original_images_subdir in sets.items():
            original_images_dir = os.path.join(base_root_dir, original_images_subdir)
            mask_images_dir = os.path.join(base_root_dir, f"JPEGImages_PSeg_GT_Mask")
            normal_images_dir = os.path.join(base_root_dir, f"JPEGImages_{set_name}_norm")

            os.makedirs(normal_images_dir, exist_ok=True)
            print(f"Generating normal images for {set_name} set in: {normal_images_dir}")

            image_files = [f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in tqdm(image_files, desc=f"Generating normal images for {set_name}"):
                image_path = os.path.join(original_images_dir, image_file)
                mask_path = os.path.join(mask_images_dir, image_file)

                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path)

                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                if mask is None:
                    print(f"Warning: Could not load mask {mask_path}")
                    continue

                # Ensure greyscale
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                binary_mask = np.zeros_like(mask, dtype=np.uint8)
                binary_mask[mask > 128] = 1 # Mark anomaly regions

                # Dilate the mask to ensure complete coverage of the anomaly
                dilation_kernel = np.ones((35, 35), np.uint8)
                dilated_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1) 
                binary_mask = (dilated_mask > 0).astype(np.uint8) # Convert back to binary (0 or 1)

                normal_image = self.apply_mask(image, binary_mask.copy())

                normal_image_path = os.path.join(normal_images_dir, image_file)
                cv2.imwrite(normal_image_path, normal_image)

        print("Normal image generation complete for all sets.")
