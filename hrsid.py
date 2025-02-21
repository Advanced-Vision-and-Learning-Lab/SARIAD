import json, glob, os, cv2, gdown
import numpy as np
import matplotlib.pyplot as plt
from anomalib.data import Folder
from anomalib import TaskType
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from absl import flags

project_root = os.path.dirname(os.path.abspath(__file__))
DRIVE_FILE_ID = "1idg_k6ccHMBsgvj86zCKUePIjGLUuHBs"

def fetch_plmstar_blob(drive_file_id):
    """
    Fetches the HRSID blob from Google Drive if it does not already exist locally.
    """
    blob_path = os.path.join(project_root, "datasets/HRSID")
    if not os.path.exists(blob_path):
        print("HRSID dataset not found locally. Downloading...")
        output_path = f"{blob_path}.zip"
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", output_path, quiet=False)
        
        import zipfile
        print("Unzipping...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall("datasets/")
        os.remove(output_path)
        print(f"Downloaded and extracted HRSID dataset to {blob_path}.")
    else:
        print("HRSID dataset found locally.")

class HRSID(Folder):
    def __init__(self, is_train=True, task=TaskType.SEGMENTATION):
        self.image_root = 'datasets/HRSID'
        self.is_train = is_train
        self.s = 'train' if self.is_train else 'test'
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.image_size=(800,800)

        fetch_plmstar_blob(f"{DRIVE_FILE_ID}")

        super().__init__(
            name="HRSID",
            root=f"./datasets/HRSID/",
            mask_dir=f"{self.s}_masks",
            normal_dir=f"{self.s}_norm",
            abnormal_dir=f"{self.s}_images",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )
        self.setup()

# SHOULD BE RUN FOR TESTING ONLY
if __name__ == "__main__":
    HRSID(is_train=False)
