from anomalib.data import Folder
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.config import DATASETS_PATH, DEBUG

NAME = "SARDet_100K"
KAGGLE = "greatbird/sardet-100k"

class SAMPLE_PUBLIC(Folder):
    def __init__(self, split="train"):
        self.split = split
        self.train_batch_size = 1 if DEBUG else 32
        self.eval_batch_size = 1 if DEBUG else 16
        self.image_size=(0,0)

        fetch_blob(NAME, kaggle=KAGGLE)

        super().__init__(
            name = NAME,
            root = f"{DATASETS_PATH}/{NAME}/",
            mask_dir = f"",
            normal_dir = f"",
            abnormal_dir = f"",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
        )

        self.setup()
