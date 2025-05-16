from anomalib.data import Folder
from anomalib import TaskType
from SARIAD.data import fetch_blob
from SARIAD import DATASETS_PATH

dataset_name = "HRSID"
DRIVE_FILE_ID = "1idg_k6ccHMBsgvj86zCKUePIjGLUuHBs"

class HRSID(Folder):
    def __init__(self, split="train", task=TaskType.SEGMENTATION):
        self.split = split
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.image_size=(800,800)

        fetch_blob(dataset_name, drive_file_id=DRIVE_FILE_ID)

        super().__init__(
            name="HRSID",
            root=f"{DATASETS_PATH}/HRSID/",
            mask_dir=f"{self.split}_masks",
            normal_dir=f"{self.split}_norm",
            abnormal_dir=f"{self.split}_images",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )

        self.setup()
