import os, gdown, zipfile, rarfile
from SARIAD import DATASETS_PATH

def fetch_blob(drive_file_id, dataset_name, datasets_dir=DATASETS_PATH, ext="zip"):
    """
    Fetches the dataset_name blob from Google Drive if it does not already exist locally.
    """
    blob_path = os.path.join(datasets_dir, dataset_name)

    if not os.path.exists(blob_path):
        print(f"{dataset_name} dataset not found locally. Downloading...")
        os.makedirs(datasets_dir, exist_ok=True)
        output_path = f"{blob_path}.{ext}"

        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", output_path, quiet=False)

        print(f"Extracting the {ext} archive...")
        if ext == "zip":
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(datasets_dir)
        elif ext == "rar":
            with rarfile.RarFile(output_path) as rar_ref:
                rar_ref.extractall(datasets_dir)
        os.remove(output_path)

        print(f"Downloaded and extracted {dataset_name} dataset to {blob_path}.")
    else:
        print(f"{dataset_name} dataset found locally.")
