from SARIAD.config import DATASETS_PATH

import os, shutil, requests, zipfile, tarfile, rarfile, gdown, kagglehub
from tqdm import tqdm

def fetch_blob(dataset_name, link="", drive_file_id="", kaggle="", ext="zip", datasets_dir=DATASETS_PATH):
    """
    Fetches the dataset_name blob from a direct link, Google Drive, or Kaggle.
    
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
    """
    Extracts an archive file to a specified directory, showing progress with tqdm.

    Parameters:
        archive_path (str): The path to the archive file.
        extract_to (str): The directory where the archive contents will be extracted.
        ext (str): The extension of the archive file (e.g., "zip", "rar", "tar.gz").

    Raises:
        ValueError: If the archive extension is not supported.
    """
    os.makedirs(extract_to, exist_ok=True)

    if ext == "zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get list of members for tqdm total
            members = zip_ref.namelist()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                zip_ref.extract(member, extract_to)
    elif ext == "rar":
        with rarfile.RarFile(archive_path) as rar_ref:
            # rarfile.infolist() gives info for tqdm total
            members = rar_ref.infolist()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                rar_ref.extract(member, extract_to)
    elif ext == "tar.gz":
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            # tarfile.getmembers() gives info for tqdm total
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                tar_ref.extract(member, extract_to)
    else:
        raise ValueError(f"Unsupported archive extension: {ext}")
