from anomalib.data import Folder
from anomalib import TaskType
from SARIAD.data import fetch_blob
from SARIAD import DATASETS_PATH

NAME = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"

class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", split="train", task=TaskType.SEGMENTATION):
        self.split = split
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.image_size=(512,512)

        fetch_blob(NAME, drive_file_id=DRIVE_FILE_ID, ext="rar")

        super().__init__(
            name=NAME,
            root=f"{DATASETS_PATH}/{NAME}/{sub_dataset}",
            mask_dir=f"voc_style/JPEGImages_PSeg_GT_Mask",
            normal_dir=f"DOESNT_YET_EXIST", # Need to generate
            abnormal_dir=f"voc_style/JPEGImages_{self.split}{'_' + sub_category if sub_category != '' else ''}",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )

        self.setup()
