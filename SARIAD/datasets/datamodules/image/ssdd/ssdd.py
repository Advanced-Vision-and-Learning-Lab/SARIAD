import os, cv2, shutil
import numpy as np
from tqdm import tqdm

from anomalib.data import Folder
from SARIAD.config import DATASETS_PATH, DEBUG
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.utils.normal_gen import generate_normal


NAME = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"


class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", split="train", batch_size=16,
                 num_workers=8, path=None, **folder_kwargs):
        self.split = split
        self.dataset_root = path or os.path.join(DATASETS_PATH, NAME)
        self.train_batch_size = 1 if DEBUG else batch_size
        self.eval_batch_size = 1 if DEBUG else batch_size
        # self.image_size = (512, 512)

        fetch_blob(self.dataset_root, drive_file_id=DRIVE_FILE_ID, ext="rar")
        if (not os.path.exists(os.path.join(self.dataset_root, sub_dataset, "train"))) \
            or (not os.path.exists(os.path.join(self.dataset_root, sub_dataset, "test"))):
            self.split_masks()
            self.generate_norm()
            self.restructure_dataset()

        super().__init__(
            name=NAME,
            root=os.path.join(self.dataset_root, sub_dataset),
            mask_dir=f"train/masks",
            normal_dir=f"train/norm",
            abnormal_dir=f"train/anom",
            normal_test_dir=f"test/norm",
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=num_workers,
            **folder_kwargs,
        )

    def restructure_dataset(self):
        base_root_dir = f"{self.dataset_root}/PSeg_SSDD"
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
        base_root_dir = f"{self.dataset_root}/PSeg_SSDD/voc_style"

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

    def generate_norm(self):
        base_root_dir = f"{self.dataset_root}/PSeg_SSDD/voc_style"

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

                binary_mask = (mask > 128).astype(np.uint8)
                normal_image = generate_normal(image, binary_mask, method="patch", dilate=35)

                normal_image_path = os.path.join(normal_images_dir, image_file)
                cv2.imwrite(normal_image_path, normal_image)

        print("Normal image generation complete for all sets.")
