import os, gdown, rarfile
from anomalib.data import Folder
from anomalib import TaskType

project_root = os.getcwd()
dataset_name = "Official-SSDD-OPEN"
DRIVE_FILE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"

def fetch_blob(drive_file_id):
    """
    Fetches the SSDD blob from Google Drive if it does not already exist locally.
    """
    datasets_dir = os.path.join(project_root, "datasets")
    blob_path = os.path.join(datasets_dir, dataset_name)

    if not os.path.exists(blob_path):
        print(f"{dataset_name} dataset not found locally. Downloading...")
        os.makedirs(datasets_dir, exist_ok=True)
        output_path = f"{blob_path}.rar"

        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", output_path, quiet=False)

        print("Extracting .rar archive...")
        with rarfile.RarFile(output_path) as rar_ref:
            rar_ref.extractall(datasets_dir)
        os.remove(output_path)
        print(f"Downloaded and extracted {dataset_name} dataset to {blob_path}.")
    else:
        print(f"{dataset_name} dataset found locally.")

class SSDD(Folder):
    def __init__(self, sub_dataset="PSeg_SSDD", sub_category="", is_train=True, task=TaskType.SEGMENTATION):
        self.image_root = 'datasets/' + dataset_name
        self.is_train = is_train
        self.s = 'train' if self.is_train else 'test'
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.image_size=(512,512)

        fetch_blob(f"{DRIVE_FILE_ID}")

        super().__init__(
            name=dataset_name,
            root=f"./datasets/{dataset_name}/{sub_dataset}",
            mask_dir=f"voc_style/JPEGImages_PSeg_GT_Mask",
            normal_dir=f"DOESNT_YET_EXIST", # Need to generate
            abnormal_dir=f"voc_style/JPEGImages_{self.s}{'_' + sub_category if sub_category != '' else ''}",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )
        self.setup()
