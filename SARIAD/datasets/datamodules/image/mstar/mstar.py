from anomalib.data import Folder
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.config import PROJECT_ROOT, DATASETS_PATH, DEBUG

import json, glob, os, cv2
import numpy as np
from SARIAD.utils.normal_gen import remove_target
from . import mstar_importer
import logging

logger = logging.getLogger(__name__)

NAME = "PLMSTAR"
DRIVE_FILE_ID = "1TT3SrDMW8ICcknoAXXZLLCLk0X6L1nAL"

class MSTAR(Folder):
    def __init__(self, collection='soc', split="train", target_filter=None, batch_size=32,
                 num_workers=8, path=None, segmenter="classical", **folder_kwargs):
        """
        Args:
            collection: MSTAR collection to use (e.g. ``soc``).
            split: Split used when generating the normal/mask data.
            target_filter: Optional list of target names to keep.
            batch_size: Train/eval batch size (forced to 1 when DEBUG is set).
            num_workers: Dataloader workers.
            path: Dataset directory (defaults to ``<DATASETS_PATH>/PLMSTAR``).
            segmenter: Backend that finds the target when generating normal images (see ``SARIAD.utils.normal_gen``).
            **folder_kwargs: Any other ``anomalib.data.Folder`` argument (seed, split modes, ...).
        """
        self.dataset = collection
        self.segmenter = segmenter
        self.image_root = path or os.path.join(DATASETS_PATH, NAME)
        self.split = split
        self.chip_size = 100
        self.patch_size = 100
        self.use_phase = False
        self.train_batch_size = 1 if DEBUG else batch_size
        self.eval_batch_size = 1 if DEBUG else batch_size
        self.target_filter = target_filter
        self.output_root = os.path.join(self.image_root, self.dataset, self.split)
        self.image_size = (128,128)

        fetch_blob(self.image_root, drive_file_id=DRIVE_FILE_ID)

        # Check if the main directory exists; if not, generate the dataset
        if not os.path.exists(self.output_root):
            self.generate()

        super().__init__(
            name = NAME,
            root = os.path.join(self.image_root, self.dataset),
            mask_dir = f"{self.split}/masks",
            normal_dir = f"{self.split}/norm",
            abnormal_dir = f"{self.split}/anom",
            normal_test_dir = "test/norm",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
            num_workers = num_workers,
            **folder_kwargs,
        )

    def data_scaling(self, chip):
        r = chip.max() - chip.min()
        return (chip - chip.min()) / r

    def log_scale(self, chip):
        return np.log10(np.abs(chip) + 1)

    def generate_cat(self, src_path, anom_dir, norm_dir, mask_dir, json_dir, split, chip_size, patch_size, use_phase, dataset):
        if not os.path.exists(src_path):
            logger.info(f'{src_path} does not exist')
            return

        category_name = os.path.basename(src_path)

        # Create category-specific subdirectories
        category_anom_dir = os.path.join(anom_dir, category_name)
        category_norm_dir = os.path.join(norm_dir, category_name)
        category_mask_dir = os.path.join(mask_dir, category_name)
        category_json_dir = os.path.join(json_dir, category_name)

        for directory in [category_anom_dir, category_norm_dir, category_mask_dir, category_json_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"Processing category: {category_name}")
        _mstar = mstar_importer.MSTAR(
            name=dataset, split=split, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
        )

        # List of source images
        image_list = glob.glob(os.path.join(src_path, '*'))

        # Process each image
        for path in image_list:
            label, _image = _mstar.read(path)
            i = 0
            # for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]

            # Save JSON metadata
            with open(os.path.join(category_json_dir, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            # Save the image with proper casting
            _image = np.nan_to_num(_image, nan=0.0, posinf=0.0, neginf=0.0)  # Remove invalid values
            cv2.imwrite(os.path.join(category_anom_dir, f'{name}-{i}.png'), (_image * 255).astype(np.uint8))

            # Remove the target and its shadow to get the normal image, and keep the removed area as the mask
            shadow_sigma = 5 if "ZIL131" in category_anom_dir else 10
            normal, combined_mask = remove_target(_image, segmenter=self.segmenter, shadow_sigma=shadow_sigma)

            cv2.imwrite(os.path.join(category_mask_dir, f'{name}-{i}.png'), combined_mask*255)
            cv2.imwrite(os.path.join(category_norm_dir, f'{name}-{i}.png'), (normal * 255).astype(np.uint8))

    def generate(self):
        dataset_root = os.path.join(PROJECT_ROOT, self.image_root, self.dataset)
        raw_root = os.path.join(dataset_root, 'raw')
        output_root = os.path.join(dataset_root, self.split)

        # Create overall directories for `anom`, `norm`, `masks`, and `json`
        anom_dir = os.path.join(output_root, 'anom')
        norm_dir = os.path.join(output_root, 'norm')
        mask_dir = os.path.join(output_root, 'masks')
        json_dir = os.path.join(output_root, 'json')

        for folder in [anom_dir, norm_dir, mask_dir, json_dir]:
            os.makedirs(folder, exist_ok=True)

        # Filter targets if a target_filter is specified
        target_list = mstar_importer.target_name[self.dataset]
        if self.target_filter:
            target_list = [target for target in target_list if target in self.target_filter]

        # Process each target category
        for target in target_list:
            self.generate_cat(
                src_path=os.path.join(raw_root, self.split, target),
                anom_dir=anom_dir,
                norm_dir=norm_dir,
                mask_dir=mask_dir,
                json_dir=json_dir,
                split=self.split,
                chip_size=self.chip_size,
                patch_size=self.patch_size,
                use_phase=self.use_phase,
                dataset=self.dataset,
            )
