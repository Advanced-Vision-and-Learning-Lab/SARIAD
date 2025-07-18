import os, shutil, requests, zipfile, tarfile, rarfile, gdown, kagglehub
from tqdm import tqdm

from SARIAD.config import DATASETS_PATH

def fetch_blob(path, link="", drive_file_id="", kaggle="", ext="zip"):
    """
    Fetches the dataset blob from a direct link, Google Drive, or Kaggle,
    and extracts it directly to the specified path.

    Parameters:
    - path: str, The full path to the directory where the extracted blob should reside.
    - link: str, optional, direct HTTP(s) link to an archive.
    - drive_file_id: str, optional, ID for Google Drive file (archive).
    - kaggle: str, optional, KaggleHub dataset slug.
    - ext: str, archive type (zip, tar.gz, rar, tar), used for link and drive_file_id.
           This parameter is ignored if 'kaggle' is provided.
    """
    if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
        print(f"Dataset found locally at: {path}")
        return

    print(f"Dataset not found locally at {path}. Downloading...")
    os.makedirs(path, exist_ok=True)

    if link:
        temp_archive_name = f"{os.path.basename(path)}_archive.{ext}"
        temp_archive_path = os.path.join(os.path.dirname(path) or '.', temp_archive_name)
        
        response = requests.get(link, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download file from {link}: HTTP {response.status_code}")
        
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(temp_archive_path)}")
        
        with open(temp_archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

        print(f"Extracting the {ext} archive...")
        _extract_archive(temp_archive_path, path, ext)
        os.remove(temp_archive_path)
        print(f"Downloaded and extracted to {path}.")

    elif drive_file_id:
        temp_archive_name = f"{os.path.basename(path)}_archive.{ext}"
        temp_archive_path = os.path.join(os.path.dirname(path) or '.', temp_archive_name)
        
        print(f"Downloading from Google Drive ID: {drive_file_id}")
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", temp_archive_path, quiet=False)
        
        print(f"Extracting the {ext} archive...")
        _extract_archive(temp_archive_path, path, ext)
        os.remove(temp_archive_path)
        print(f"Downloaded and extracted to {path}.")

    elif kaggle:
        downloaded_kaggle_path = kagglehub.dataset_download(kaggle)
        print(f"KaggleHub {kaggle} dataset downloaded to: {downloaded_kaggle_path}")
        
        os.makedirs(path, exist_ok=True)

        for item in os.listdir(downloaded_kaggle_path):
            s = os.path.join(downloaded_kaggle_path, item)
            d = os.path.join(path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        print(f"KaggleHub {kaggle} dataset copied to: {path}")

    else:
        raise ValueError("Must provide either a `link`, `drive_file_id`, or `kaggle` slug.")

def _extract_archive(archive_path, extract_to, ext):
    """
    Extracts an archive file to a specified directory, showing progress with tqdm.
    It handles cases where the archive contains a single top-level directory,
    moving its contents directly to 'extract_to' to avoid nested folders.

    Parameters:
        archive_path (str): The path to the archive file.
        extract_to (str): The directory where the archive contents will be extracted.
        ext (str): The extension of the archive file (e.g., "zip", "rar", "tar.gz").

    Raises:
        ValueError: If the archive extension is not supported.
    """
    temp_extract_dir = f"{archive_path}_temp_extracted"
    os.makedirs(temp_extract_dir, exist_ok=True)

    if ext == "zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                zip_ref.extract(member, temp_extract_dir)
    elif ext == "rar":
        with rarfile.RarFile(archive_path) as rar_ref:
            members = rar_ref.infolist()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                rar_ref.extract(member, temp_extract_dir)
    elif ext == "tar.gz":
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                tar_ref.extract(member, temp_extract_dir)
    elif ext == "tar":
        with tarfile.open(archive_path, 'r:') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {os.path.basename(archive_path)}"):
                tar_ref.extract(member, temp_extract_dir)
    else:
        raise ValueError(f"Unsupported archive extension: {ext}. Supported: zip, rar, tar, tar.gz")

    # After extraction to temp_extract_dir, move contents to final extract_to
    extracted_contents = os.listdir(temp_extract_dir)

    if len(extracted_contents) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_contents[0])):
        # If there's a single directory, assume it's the redundant top-level folder
        source_dir = os.path.join(temp_extract_dir, extracted_contents[0])
        for item in os.listdir(source_dir):
            shutil.move(os.path.join(source_dir, item), extract_to)
        shutil.rmtree(temp_extract_dir)
    else:
        # Otherwise, move all contents directly
        for item in extracted_contents:
            shutil.move(os.path.join(temp_extract_dir, item), extract_to)
        shutil.rmtree(temp_extract_dir)

def fetch_dataset(dataset_name, datasets_dir=DATASETS_PATH, link="", drive_file_id="", kaggle="", ext="zip"):
    """
    Fetches a dataset blob from a direct link, Google Drive, or Kaggle,
    maintaining backward compatibility with the original fetch_blob signature.

    Parameters:
    - dataset_name: str, The name of the dataset. This will be the directory name inside datasets_dir.
    - datasets_dir: str, The root directory where datasets are stored.
    - link: str, optional, direct HTTP(s) link to an archive.
    - drive_file_id: str, optional, ID for Google Drive file (archive).
    - kaggle: str, optional, KaggleHub dataset slug.
    - ext: str, archive type (zip, tar.gz, rar, tar), used for link and drive_file_id.
           This parameter is ignored if 'kaggle' is provided.
    """
    full_dataset_path = os.path.join(datasets_dir, dataset_name)
    fetch_blob(
        path=full_dataset_path,
        link=link,
        drive_file_id=drive_file_id,
        kaggle=kaggle,
        ext=ext
    )
