from anomalib.data import Folder
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.config import DATASETS_PATH, DEBUG

import os

NAME = "HRSID"
DRIVE_FILE_ID = "1idg_k6ccHMBsgvj86zCKUePIjGLUuHBs"

class HRSID(Folder):
    def __init__(self, split="train", batch_size=16, num_workers=8, path=None, **folder_kwargs):
        self.split = split
        self.dataset_root = path or os.path.join(DATASETS_PATH, NAME)
        self.train_batch_size = 1 if DEBUG else batch_size
        self.eval_batch_size = 1 if DEBUG else batch_size
        self.image_size = (800,800)

        fetch_blob(self.dataset_root, drive_file_id=DRIVE_FILE_ID)

        super().__init__(
            name = NAME,
            root = self.dataset_root,
            mask_dir = f"{self.split}_masks",
            normal_dir = f"{self.split}_norm",
            abnormal_dir = f"{self.split}_images",
            normal_test_dir = "test_good",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
            num_workers = num_workers,
            **folder_kwargs,
        )
