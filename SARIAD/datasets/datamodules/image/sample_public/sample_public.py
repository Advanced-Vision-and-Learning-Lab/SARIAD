from anomalib.data import Folder
from anomalib import TaskType
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.config import DATASETS_PATH

NAME = "SAMPLE_dataset_public"
LINK = "https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/archive/refs/heads/master.zip"

class SAMPLE_PUBLIC(Folder):
    def __init__(self, split="train", task=TaskType.SEGMENTATION):
        self.split = split
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.image_size=(0,0)

        fetch_blob(NAME, link=LINK)

        super().__init__(
            name=NAME,
            root=f"{DATASETS_PATH}/{NAME}/",
            mask_dir=f"",
            normal_dir=f"",
            abnormal_dir=f"",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )

        self.setup()
