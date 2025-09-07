import os, cv2, random, shutil
import numpy as np
from tqdm import tqdm

from anomalib.data import Folder
from SARIAD.config import DATASETS_PATH, DEBUG
from SARIAD.utils.blob_utils import fetch_dataset
from SARIAD.utils.img_utils import img_debug
from SARIAD.pre_processing.SARCNN import *


NAME = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"


class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", split="train", batch_size=16):
        self.split = split
        self.train_batch_size = 1 if DEBUG else batch_size
        self.eval_batch_size = 1 if DEBUG else batch_size
        # self.image_size = (512, 512)

        fetch_dataset(NAME, drive_file_id=DRIVE_FILE_ID, ext="rar")
        if (not os.path.exists(os.path.join(DATASETS_PATH, NAME, sub_dataset, "train"))) \
            or (not os.path.exists(os.path.join(DATASETS_PATH, NAME, sub_dataset, "test"))):
            self.split_masks()
            self.generate_norm()
            self.restructure_dataset()

        super().__init__(
            name=NAME,
            root=os.path.join(DATASETS_PATH, NAME, sub_dataset),
            mask_dir=f"train/masks",
            normal_dir=f"train/norm",
            abnormal_dir=f"train/anom",
            normal_test_dir=f"test/norm",
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
        )

        self.setup()

    def restructure_dataset(self):
        base_root_dir = f"{DATASETS_PATH}/{NAME}/PSeg_SSDD"
        voc_style_dir = os.path.join(base_root_dir, "voc_style")

        target_dir = base_root_dir

        if os.path.exists(os.path.join(target_dir, "train", "anom")):
            print("Dataset already restructured. Skipping.")
            return

        print("Restructuring dataset...")

        os.makedirs(os.path.join(target_dir, "train", "anom"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "test", "anom"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "train", "masks"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "test", "masks"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "train", "norm"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "test", "norm"), exist_ok=True)

        for split in ["train", "test"]:
            anom_source_dir = os.path.join(voc_style_dir, f"JPEGImages_{split}")
            mask_source_dir = os.path.join(
                voc_style_dir, f"JPEGImages_PSeg_GT_Mask_{split}"
            )
            norm_source_dir = os.path.join(voc_style_dir, f"JPEGImages_{split}_norm")

            anom_dest_dir = os.path.join(target_dir, split, "anom")
            mask_dest_dir = os.path.join(target_dir, split, "masks")
            norm_dest_dir = os.path.join(target_dir, split, "norm")

            for filename in os.listdir(anom_source_dir):
                shutil.copy(os.path.join(anom_source_dir, filename), anom_dest_dir)

            if os.path.exists(mask_source_dir):
                for filename in os.listdir(mask_source_dir):
                    source_path = os.path.join(mask_source_dir, filename)
                    dest_path = os.path.join(mask_dest_dir, filename)
                    
                    # Load the mask
                    mask = cv2.imread(source_path)
                    if mask is not None:
                        # Convert green-on-black to white-on-black
                        green_channel = mask[:, :, 1]
                        _, binary_mask = cv2.threshold(green_channel, 1, 255, cv2.THRESH_BINARY)
                        # Save the new binary mask
                        cv2.imwrite(dest_path, binary_mask)
                    else:
                        print(f"Warning: Could not read mask at {source_path}. Skipping.")

            if os.path.exists(norm_source_dir):
                for filename in os.listdir(norm_source_dir):
                    shutil.copy(os.path.join(norm_source_dir, filename), norm_dest_dir)

        shutil.rmtree(voc_style_dir)
        print("Dataset restructuring complete.")

    def split_masks(self):
        base_root_dir = f"{DATASETS_PATH}/{NAME}/PSeg_SSDD/voc_style"

        source_mask_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask")
        train_masks_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask_train")
        test_masks_dir = os.path.join(base_root_dir, "JPEGImages_PSeg_GT_Mask_test")

        original_train_images_dir = os.path.join(base_root_dir, "JPEGImages_train")
        original_test_images_dir = os.path.join(base_root_dir, "JPEGImages_test")

        if (
            os.path.exists(train_masks_dir)
            and os.listdir(train_masks_dir)
            and os.path.exists(test_masks_dir)
            and os.listdir(test_masks_dir)
        ):
            print("Masks are already split. Skipping mask splitting.")
            return

        print("Splitting masks into train and test directories.")

        os.makedirs(train_masks_dir, exist_ok=True)
        os.makedirs(test_masks_dir, exist_ok=True)

        train_image_files = {
            f
            for f in os.listdir(original_train_images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        }
        test_image_files = {
            f
            for f in os.listdir(original_test_images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        }

        mask_files = [
            f
            for f in os.listdir(source_mask_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        for mask_file in tqdm(mask_files, desc="Splitting masks"):
            source_mask_path = os.path.join(source_mask_dir, mask_file)

            if mask_file in train_image_files:
                destination_mask_path = os.path.join(train_masks_dir, mask_file)
            elif mask_file in test_image_files:
                destination_mask_path = os.path.join(test_masks_dir, mask_file)
            else:
                print(
                    f"Warning: Mask {mask_file} does not correspond to any image in train or test sets. Skipping."
                )
                continue
            shutil.copy2(source_mask_path, destination_mask_path)

        print("Mask splitting complete.")

    def apply_mask(self, image, mask, min_crop_size=3, max_crop_size=15, sample_step=10):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_float = image.astype(np.float32)

        mask_int = mask.astype(np.uint8)
        indices_not_in_mask = np.where(mask_int == 0)
        img_mean = image_float[indices_not_in_mask].mean()

        comp = (
            lambda x: (x >= img_mean) if (img_mean / 255.0 >= 0.5) else (x <= img_mean)
        )

        h, w = mask.shape
        working_mask = mask.copy()

        patch_pool = []

        for y_top in range(0, h - min_crop_size + 1, sample_step):
            for x_left in range(0, w - min_crop_size + 1, sample_step):
                if y_top + max_crop_size <= h and x_left + max_crop_size <= w:
                    if np.all(
                        mask[
                            y_top : y_top + max_crop_size,
                            x_left : x_left + max_crop_size,
                        ]
                        == 0
                    ):
                        for k_size in range(min_crop_size, max_crop_size + 1):
                            current_patch = image_float[
                                y_top : y_top + k_size, x_left : x_left + k_size
                            ]
                            if (
                                current_patch.shape[0] == k_size
                                and current_patch.shape[1] == k_size
                            ):
                                if comp(current_patch.mean()):
                                    patch_pool.append(current_patch)
                else:
                    for k_size in range(min_crop_size, max_crop_size + 1):
                        if y_top + k_size <= h and x_left + k_size <= w:
                            current_patch = image_float[
                                y_top : y_top + k_size, x_left : x_left + k_size
                            ]
                            if (
                                np.all(
                                    mask[
                                        y_top : y_top + k_size,
                                        x_left : x_left + k_size,
                                    ]
                                    == 0
                                )
                                and current_patch.shape[0] == k_size
                                and current_patch.shape[1] == k_size
                            ):
                                if comp(current_patch.mean()):
                                    patch_pool.append(current_patch)

        if not patch_pool:
            print("Warning: No valid background patches found for filling. Returning original image.")
            return image

        masked_y_coords, masked_x_coords = np.where(mask == 1)

        masked_pixels = list(zip(masked_y_coords, masked_x_coords))
        random.shuffle(masked_pixels)

        temp_fill_canvas = np.zeros_like(image_float)
        temp_weight_canvas = np.zeros_like(image_float, dtype=np.int32)

        for cy, cx in masked_pixels:
            if working_mask[cy, cx] == 0:
                continue

            patch_to_paste = random.choice(patch_pool)
            ph, pw = patch_to_paste.shape

            target_y_start = cy - ph // 2
            target_x_start = cx - pw // 2

            ty_start = max(0, target_y_start)
            tx_start = max(0, target_x_start)
            ty_end = min(h, target_y_start + ph)
            tx_end = min(w, target_x_start + pw)

            patch_slice_y_start = 0 if target_y_start >= 0 else -target_y_start
            patch_slice_x_start = 0 if target_x_start >= 0 else -target_x_start

            current_patch_portion = patch_to_paste[
                patch_slice_y_start : patch_slice_y_start + (ty_end - ty_start),
                patch_slice_x_start : patch_slice_x_start + (tx_end - tx_start),
            ]

            mask_region_for_paste = working_mask[ty_start:ty_end, tx_start:tx_end]

            if mask_region_for_paste.size > 0 and np.any(mask_region_for_paste == 1):
                indices_to_fill_y, indices_to_fill_x = np.where(
                    mask_region_for_paste == 1
                )
                temp_fill_canvas[ty_start:ty_end, tx_start:tx_end][
                    indices_to_fill_y, indices_to_fill_x
                ] += current_patch_portion[indices_to_fill_y, indices_to_fill_x]
                temp_weight_canvas[ty_start:ty_end, tx_start:tx_end][
                    indices_to_fill_y, indices_to_fill_x
                ] += 1
                working_mask[ty_start:ty_end, tx_start:tx_end][
                    indices_to_fill_y, indices_to_fill_x
                ] = 0

        temp_weight_canvas[temp_weight_canvas == 0] = 1
        filled_region_avg = temp_fill_canvas / temp_weight_canvas

        final_output_image = image_float.copy()
        final_output_image[mask == 1] = filled_region_avg[mask == 1]

        return np.clip(final_output_image, 0, 255).astype(image.dtype)

    def generate_norm(self):
        base_root_dir = f"{DATASETS_PATH}/{NAME}/PSeg_SSDD/voc_style"

        sets = {"train": "JPEGImages_train", "test": "JPEGImages_test"}

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

            image_files = [f for f in os.listdir(original_images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
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

                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                binary_mask = np.zeros_like(mask, dtype=np.uint8)
                binary_mask[mask > 128] = 1

                dilation_kernel = np.ones((35, 35), np.uint8)
                dilated_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)
                binary_mask = (dilated_mask > 0).astype(np.uint8)

                normal_image = self.apply_mask(image, binary_mask.copy())

                normal_image_path = os.path.join(normal_images_dir, image_file)
                cv2.imwrite(normal_image_path, normal_image)

        print("Normal image generation complete for all sets.")
