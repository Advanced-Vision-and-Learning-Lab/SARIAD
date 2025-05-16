import os, shutil, requests, zipfile, tarfile, rarfile, gdown, kagglehub
from SARIAD import DATASETS_PATH

def fetch_blob(dataset_name, link="", drive_file_id="", kaggle="", ext="zip", datasets_dir=DATASETS_PATH):
    """
    Fetches the dataset_name blob from a GitHub repo, direct link, Google Drive, or Kaggle.
    
    Parameters:
    - dataset_name: str, name of the dataset directory
    - link: str, optional, direct HTTP(s) link
    - drive_file_id: str, optional, ID for Google Drive file
    - kaggle: str, optional, KaggleHub dataset slug
    - ext: str, archive type (zip, tar.gz, rar)
    - datasets_dir: str, path to local datasets directory
    """
    blob_path = os.path.join(datasets_dir, dataset_name)

    if os.path.exists(blob_path):
        print(f"{dataset_name} dataset found locally.")
        return

    print(f"{dataset_name} dataset not found locally. Downloading...")
    os.makedirs(datasets_dir, exist_ok=True)

    if link:
        archive_path = f"{blob_path}.{ext}"
        response = requests.get(link, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download file from {link}: HTTP {response.status_code}")
        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Extracting the {ext} archive...")
        _extract_archive(archive_path, datasets_dir, ext)
        os.remove(archive_path)
        print(f"Downloaded and extracted {dataset_name} to {blob_path}.")

    elif drive_file_id:
        archive_path = f"{blob_path}.{ext}"
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", archive_path, quiet=False)
        print(f"Extracting the {ext} archive...")
        _extract_archive(archive_path, datasets_dir, ext)
        os.remove(archive_path)
        print(f"Downloaded and extracted {dataset_name} to {blob_path}.")

    elif kaggle:
        path = kagglehub.dataset_download(kaggle)
        shutil.copytree(path, blob_path)
        print(f"KaggleHub {kaggle} dataset copied to: {blob_path}")

    else:
        raise ValueError("Must provide either a `link` or `drive_file_id`.")

def _extract_archive(archive_path, extract_to, ext):
    if ext == "zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif ext == "rar":
        with rarfile.RarFile(archive_path) as rar_ref:
            rar_ref.extractall(extract_to)
    elif ext == "tar.gz":
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive extension: {ext}")
