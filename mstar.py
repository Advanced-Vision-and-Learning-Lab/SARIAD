import json, glob, os, cv2, mstar_importer
import numpy as np
import matplotlib.pyplot as plt
from anomalib.data import Folder
from anomalib import TaskType
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from absl import flags
project_root = os.path.dirname(os.path.abspath(__file__))

class MSTAR(Folder):
    def __init__(self, collection='soc', is_train=True, task=TaskType.CLASSIFICATION):
        self.dataset = collection
        self.image_root = 'datasets/PLMSTAR'
        self.is_train = is_train
        self.s = 'train' if self.is_train else 'test'
        self.chip_size = 100
        self.patch_size = 100
        self.use_phase = False
        self.output_root = os.path.join(self.image_root, self.dataset, self.s)

        # Check if the main directory exists; if not, generate the dataset
        if not os.path.exists(self.output_root):
            self.generate()

        super().__init__(
            name="MSTAR",
            root=f"./datasets/PLMSTAR/{self.dataset}/",
            mask_dir=f"{self.s}/masks",
            normal_dir=f"{self.s}/norm",
            abnormal_dir=f"{self.s}/anom",
            image_size=(256, 256),
            train_batch_size=32,
            eval_batch_size=32,
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
        height, width, channels = mask.shape
        for y in range(height):
            for x in range(width):
                # Check if the current position is an anomalous area
                if mask[y, x, 0] != 0:  # Assuming mask is single-channel but has been expanded
                    for c in range(channels):  # Iterate over each color channel
                        output_image[y, x, c] = np.random.normal(mean, std)

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
            label, _images = _mstar.read(path)
            for i, _image in enumerate(_images):
                name = os.path.splitext(os.path.basename(path))[0]

                # Save JSON metadata
                with open(os.path.join(category_json_dir, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                    json.dump(label, f, ensure_ascii=False, indent=2)

                cv2.imwrite(os.path.join(category_anom_dir, f'{name}-{i}.png'), (_image * 255).astype(np.uint8))

                # Generate mask and save it as PNG
                mask = self.generate_mask(_image)
                cv2.imwrite(os.path.join(category_mask_dir, f'{name}-{i}.png'), mask)

                # Generate and save normal image as PNG
                normal_image = self.apply_mask_to_image(_image, mask)
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

#SHOULD BE RUN FOR TESTING ONLY
if __name__ == "__main__":
    MSTAR(is_train=False)
