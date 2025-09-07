from anomalib.data import Folder
from SARIAD.utils.blob_utils import fetch_dataset
from SARIAD.config import DATASETS_PATH, DEBUG

import os

NAME = "SAMPLE_dataset_public"
LINK = "https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/archive/refs/heads/master.zip"

class SAMPLE_PUBLIC(Folder):
    def __init__(self, split="train", batch_size=16):
        self.split = split
        self.train_batch_size = 1 if DEBUG else batch_size
        self.eval_batch_size = 1 if DEBUG else batch_size
        self.image_size=(0,0)

        fetch_dataset(NAME, link=LINK)

        super().__init__(
            name = NAME,
            root = os.path.join(DATASETS_PATH,NAME),
            mask_dir = f"",
            normal_dir = f"",
            abnormal_dir = f"",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
        )

        self.setup()
