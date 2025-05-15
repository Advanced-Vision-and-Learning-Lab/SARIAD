from anomalib.data import Folder
from anomalib import TaskType
from SARIAD.data import fetch_blob
from SARIAD import DATASETS_PATH

dataset_name = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"

class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", is_train=True, task=TaskType.SEGMENTATION):
        self.image_root = 'datasets/' + dataset_name
        self.is_train = is_train
        self.s = 'train' if self.is_train else 'test'
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.image_size=(512,512)

        fetch_blob(DRIVE_FILE_ID, dataset_name)

        super().__init__(
            name=dataset_name,
            root=f"{DATASETS_PATH}/{dataset_name}/{sub_dataset}",
            mask_dir=f"voc_style/JPEGImages_PSeg_GT_Mask",
            normal_dir=f"DOESNT_YET_EXIST", # Need to generate
            abnormal_dir=f"voc_style/JPEGImages_{self.s}{'_' + sub_category if sub_category != '' else ''}",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )

        self.setup()
