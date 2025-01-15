import json, glob, os, cv2, mstar_importer, gdown
import numpy as np
import matplotlib.pyplot as plt
from anomalib.data import Folder
from anomalib import TaskType
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from absl import flags

project_root = os.path.dirname(os.path.abspath(__file__))
DRIVE_FILE_ID = ""

def fetch_plmstar_blob(drive_file_id):
    """
    Fetches the PLMSTAR blob from Google Drive if it does not already exist locally.
    """
    blob_path = os.path.join(project_root, "datasets/PLMSTAR")
    if not os.path.exists(blob_path):
        print("PLMSTAR dataset not found locally. Downloading...")
        output_path = f"{blob_path}.zip"
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", output_path, quiet=False)
        
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(blob_path)
        os.remove(output_path)
        print(f"Downloaded and extracted PLMSTAR dataset to {blob_path}.")
    else:
        print("PLMSTAR dataset found locally.")

class MSTAR(Folder):
    def __init__(self, collection='soc', is_train=True, task=TaskType.CLASSIFICATION, target_filter=None):
        self.dataset = collection
        self.image_root = 'datasets/PLMSTAR'
        self.is_train = is_train
        self.s = 'train' if self.is_train else 'test'
        self.chip_size = 100
        self.patch_size = 100
        self.use_phase = False
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.target_filter = target_filter
        self.output_root = os.path.join(self.image_root, self.dataset, self.s)
        self.image_size=(128,128)

        fetch_plmstar_blob(f"{DRIVE_FILE_ID}")

        # Check if the main directory exists; if not, generate the dataset
        if not os.path.exists(self.output_root):
            self.generate()

        super().__init__(
            name="MSTAR",
            root=f"./datasets/PLMSTAR/{self.dataset}/",
            mask_dir=f"{self.s}/masks",
            normal_dir=f"{self.s}/norm",
            abnormal_dir=f"{self.s}/anom",
            image_size=self.image_size,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            task=task,
        )
        self.setup()

    def generate_mask(self, image, n_clusters=2, sigma=1, kernel_size=50):
        """
        Generate a filled mask for the image using KMeans clustering,
        with Gaussian blurring and morphological operations to fill shapes.
        
        Parameters:
            image (np.array): The input image as a NumPy array.
            n_clusters (int): The number of clusters for KMeans.
            sigma (float): Standard deviation for Gaussian blur. Higher values increase blurring.
            
        Returns:
            np.array: The filled mask with cluster labels for each pixel.
        """
        # Apply Gaussian blur to reduce noise and speckling
        imagec = image.copy()
        smoothed_image = gaussian_filter(imagec, sigma=sigma)

        # Flatten the smoothed image for KMeans clustering
        reshaped_image = smoothed_image.reshape(-1, 1)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(reshaped_image)
        labels = kmeans.labels_

        # Reshape the result back to the original image shape
        mask = labels.reshape(imagec.shape)

        # Convert the mask to uint8 format for OpenCV
        mask_uint8 = (mask * (255 // (n_clusters - 1))).astype(np.uint8)

        # Apply morphological operations to fill in the shapes
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filled_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, circular_kernel)

        num_labels, labels_im = cv2.connectedComponents(filled_mask.astype(np.uint8))

        # Find the largest component by size
        largest_label = 1  # Start from 1 to skip the background (0)
        largest_size = 0
        
        for label in range(1, num_labels):
            component_size = np.sum(labels_im == label)
            if component_size > largest_size:
                largest_size = component_size
                largest_label = label

        # Create a mask that keeps only the largest component
        cleaned_mask = np.zeros_like(mask)
        cleaned_mask[labels_im == largest_label] = 255  # Keep the largest component

        return cleaned_mask

    def apply_mask_to_image(self, image, mask):
        """
        Removes anomalies from an image by filling in the masked areas with values 
        that follow the statistical distribution of surrounding pixels.
        
        Parameters:
            image (np.array): The input image with anomalies.
            mask (np.array): The mask indicating anomalous regions (non-zero values).
            neighborhood_size (int): The size of the neighborhood to sample around anomalies.
            
        Returns:
            np.array: The image with anomalies filled in.
        """
        output_image = image.copy()
        normal_area = np.where(mask == 0)
        
        # Calculate mean and std of the normal (non-anomalous) area
        normal_values = image[normal_area]
        mean, std = np.mean(normal_values), np.std(normal_values)

        output_image = image.copy()

        # Calculate mean and std of the normal (non-anomalous) area
        normal_values = image[mask == 0]
        mean, std = np.mean(normal_values), np.std(normal_values)

        # Apply changes only to masked (anomalous) areas
        height, width, _ = mask.shape  # No need to unpack channels anymore
        for y in range(height):
            for x in range(width):
                # Check if the current position is an anomalous area
                if mask[y, x] != 0:  # Assuming mask is single-channel (grayscale)
                    # Replace the pixel value with a random value from a normal distribution
                    output_image[y, x] = np.random.normal(mean, std)

        return output_image
            
    def data_scaling(self, chip):
        r = chip.max() - chip.min()
        return (chip - chip.min()) / r

    def log_scale(self, chip):
        return np.log10(np.abs(chip) + 1)

    def generate_cat(self, src_path, anom_dir, norm_dir, mask_dir, json_dir, is_train, chip_size, patch_size, use_phase, dataset):
        if not os.path.exists(src_path):
            print(f'{src_path} does not exist')
            return

        category_name = os.path.basename(src_path)

        # Create category-specific subdirectories
        category_anom_dir = os.path.join(anom_dir, category_name)
        category_norm_dir = os.path.join(norm_dir, category_name)
        category_mask_dir = os.path.join(mask_dir, category_name)
        category_json_dir = os.path.join(json_dir, category_name)

        for directory in [category_anom_dir, category_norm_dir, category_mask_dir, category_json_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"Processing category: {category_name}")
        _mstar = mstar_importer.MSTAR(
            name=dataset, is_train=is_train, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
        )

        # List of source images
        image_list = glob.glob(os.path.join(src_path, '*'))

        # Process each image
        for path in image_list:
            label, _image = _mstar.read(path)
            i = 0
            # for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]

            # Save JSON metadata
            with open(os.path.join(category_json_dir, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            # Save the image with proper casting
            _image = np.nan_to_num(_image, nan=0.0, posinf=0.0, neginf=0.0)  # Remove invalid values
            cv2.imwrite(os.path.join(category_anom_dir, f'{name}-{i}.png'), (_image * 255).astype(np.uint8))

            # Generate mask and save it as PNG
            mask = self.generate_mask(_image)
            if mask.size > 0:  # Ensure mask is not empty
                cv2.imwrite(os.path.join(category_mask_dir, f'{name}-{i}.png'), mask)
            else:
                print(f"Warning: Empty mask for image {name}-{i}")

            # Apply mask to the image and save the normal image as PNG
            normal_image = self.apply_mask_to_image(_image, mask)
            normal_image = np.nan_to_num(normal_image, nan=0.0, posinf=0.0, neginf=0.0)  # Ensure valid values
            cv2.imwrite(os.path.join(category_norm_dir, f'{name}-{i}.png'), (normal_image * 255).astype(np.uint8))

    def generate(self):
        dataset_root = os.path.join(project_root, self.image_root, self.dataset)
        raw_root = os.path.join(dataset_root, 'raw')
        mode = 'train' if self.is_train else 'test'
        output_root = os.path.join(dataset_root, mode)

        # Create overall directories for `anom`, `norm`, `masks`, and `json`
        anom_dir = os.path.join(output_root, 'anom')
        norm_dir = os.path.join(output_root, 'norm')
        mask_dir = os.path.join(output_root, 'masks')
        json_dir = os.path.join(output_root, 'json')

        for folder in [anom_dir, norm_dir, mask_dir, json_dir]:
            os.makedirs(folder, exist_ok=True)

        # Filter targets if a target_filter is specified
        target_list = mstar_importer.target_name[self.dataset]
        if self.target_filter:
            target_list = [target for target in target_list if target in self.target_filter]

        # Process each target category
        for target in mstar_importer.target_name[self.dataset]:
            self.generate_cat(
                src_path=os.path.join(raw_root, mode, target),
                anom_dir=anom_dir,
                norm_dir=norm_dir,
                mask_dir=mask_dir,
                json_dir=json_dir,
                is_train=self.is_train,
                chip_size=self.chip_size,
                patch_size=self.patch_size,
                use_phase=self.use_phase,
                dataset=self.dataset,
            )

# SHOULD BE RUN FOR TESTING ONLY
if __name__ == "__main__":
    MSTAR(is_train=False)
