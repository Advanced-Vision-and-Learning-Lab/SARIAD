from anomalib.data import Folder
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.config import PROJECT_ROOT, DATASETS_PATH, DEBUG

import json, glob, os, cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from . import mstar_importer

dataset_name = "PLMSTAR"
DRIVE_FILE_ID = "1TT3SrDMW8ICcknoAXXZLLCLk0X6L1nAL"

class MSTAR(Folder):
    def __init__(self, collection='soc', split="train", target_filter=None):
        self.dataset = collection
        self.image_root = 'datasets/PLMSTAR'
        self.split = split
        self.chip_size = 100
        self.patch_size = 100
        self.use_phase = False
        self.train_batch_size = 1 if DEBUG else 32
        self.eval_batch_size = 1 if DEBUG else 32
        self.target_filter = target_filter
        self.output_root = os.path.join(self.image_root, self.dataset, self.split)
        self.image_size = (128,128)

        fetch_blob(dataset_name, drive_file_id=DRIVE_FILE_ID)

        # Check if the main directory exists; if not, generate the dataset
        if not os.path.exists(self.output_root):
            self.generate()

        super().__init__(
            name = "MSTAR",
            root = f"{DATASETS_PATH}/PLMSTAR/{self.dataset}/",
            mask_dir = f"{self.split}/masks",
            normal_dir = f"{self.split}/norm",
            abnormal_dir = f"{self.split}/anom",
            train_batch_size = self.train_batch_size,
            eval_batch_size = self.eval_batch_size,
        )
        self.setup()

    def generate_mask(self, image, n_clusters=2, sigma=4, kernel_size=50, shadow=False):
        """
        Generate a filled mask for the image using KMeans clustering,
        with Gaussian blurring and morphological operations to fill shapes.
        
        Parameters:
            image (np.array): The input image as a NumPy array of shape (H, W, 1).
            n_clusters (int): The number of clusters for KMeans.
            sigma (float): Standard deviation for Gaussian blur. Higher values increase blurring.
            kernel_size (int): Size of the structuring element for morphological operations.
            
        Returns:
            np.array: The filled mask with the same shape as the input image.
        """
        # Check the original shape
        original_shape = image.shape

        # Apply Gaussian smoothing
        smoothed_image = gaussian_filter(image[:, :, 0], sigma=sigma)  # Remove the channel dimension temporarily

        # Flatten the smoothed image for KMeans clustering
        reshaped_image = smoothed_image.reshape(-1, 1)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100)
        kmeans.fit(reshaped_image)
        labels = kmeans.labels_

        # Identify the cluster with the lowest intensity (background)
        cluster_means = [np.mean(reshaped_image[labels == i]) for i in range(n_clusters)]
        background_label = np.argmin(cluster_means)  # The darkest cluster is the background

        # Reshape labels to the original 2D spatial shape
        labels_reshaped = labels.reshape(original_shape[:2])

        # Create a binary mask where the target (non-background) is 1
        target_mask = (labels_reshaped != background_label).astype(np.uint8)
        if(shadow):
            target_mask = 1 - target_mask

        # Morphological operations to fill in shapes
        if(kernel_size):
            circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, circular_kernel)

        # Use connected components to find and keep the largest target area
        num_labels, labels_im = cv2.connectedComponents(target_mask)

        # Find the largest connected component (excluding background, label 0)
        largest_label = 1  # Start from 1 to skip the background
        largest_size = 0

        for label in range(1, num_labels):
            component_size = np.sum(labels_im == label)
            if component_size > largest_size:
                largest_size = component_size
                largest_label = label

        # Create a mask that keeps only the largest component
        largest_component_mask = (labels_im == largest_label).astype(np.uint8)

        # Make sure the largest component is white (255) and everything else is black (0)
        final_mask = largest_component_mask  # 255 for the largest component, 0 for everything else

        # Ensure the result is in uint8 format and restore the original shape
        final_mask = final_mask.astype(np.uint8)

        return final_mask

    def apply_mask_to_image(self, image, mask):
        """
        Fills in the masked areas of the image by randomly picking pixels from the
        background (areas not covered by the mask) and assigning those values to
        the masked areas.

        Parameters:
            image (np.array): The input image with anomalies.
            mask (np.array): The mask indicating anomalous regions (non-zero values).

        Returns:
            np.array: The image with anomalies filled in.
        """

        # Create the output image by copying the original
        output_image = image.copy()

        # Get the indices of the background pixels (mask == 0)
        background_indices = np.where(mask == 0)

        # Get the background pixel values (pixels where the mask is 0)
        background_pixels = image[background_indices]

        # Check if there are any background pixels
        if len(background_pixels) == 0:
            return image

        # Iterate over the masked pixels (mask == 1)
        masked_indices = np.where(mask == 1)
        for idx in zip(*masked_indices):
            # Randomly select a background pixel
            random_pixel = background_pixels[np.random.randint(len(background_pixels))]

            # Replace the masked pixel with the selected background pixel
            output_image[idx] = random_pixel

        return output_image

            
    def data_scaling(self, chip):
        r = chip.max() - chip.min()
        return (chip - chip.min()) / r

    def log_scale(self, chip):
        return np.log10(np.abs(chip) + 1)

    def blur_mask(self, mask, sigma=1):
        """
        Apply Gaussian blur to a binary mask and threshold it back to 0 and 1.

        Parameters:
            mask (np.array): Binary mask with values 0 and 1.
            sigma (float): Standard deviation for Gaussian kernel, controls blur intensity.

        Returns:
            np.array: Blurred binary mask with values 0 and 1.
        """

        # Apply Gaussian blur on the mask
        blurred_mask = gaussian_filter(mask.astype(float), sigma=sigma)

        # Threshold the blurred mask to get values of 0 and 1
        blurred_mask = (blurred_mask > 0.15).astype(np.uint8)

        return blurred_mask

    def generate_cat(self, src_path, anom_dir, norm_dir, mask_dir, json_dir, split, chip_size, patch_size, use_phase, dataset):
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
            name=dataset, split=split, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
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
            target_mask = self.generate_mask(_image)
            target_mask = self.blur_mask(target_mask, sigma=13)
            # Apply mask to the image and save the normal image as PNG
            target_removed = self.apply_mask_to_image(_image, target_mask)
            if("ZIL131" in category_anom_dir):
                shadow_mask = self.generate_mask(target_removed, n_clusters=5, sigma=5, shadow=True, kernel_size=0)
            else:
                shadow_mask = self.generate_mask(target_removed, n_clusters=5, sigma=10, shadow=True, kernel_size=0)
            shadow_mask = self.blur_mask(shadow_mask, sigma=10)
            combined_mask = np.maximum(target_mask, shadow_mask)

            cv2.imwrite(os.path.join(category_mask_dir, f'{name}-{i}.png'), combined_mask*255)
            normal = self.apply_mask_to_image(target_removed, shadow_mask)
            normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)  # Ensure valid values

            cv2.imwrite(os.path.join(category_norm_dir, f'{name}-{i}.png'), (normal * 255).astype(np.uint8))

    def generate(self):
        dataset_root = os.path.join(PROJECT_ROOT, self.image_root, self.dataset)
        raw_root = os.path.join(dataset_root, 'raw')
        output_root = os.path.join(dataset_root, self.split)

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
                src_path=os.path.join(raw_root, self.split, target),
                anom_dir=anom_dir,
                norm_dir=norm_dir,
                mask_dir=mask_dir,
                json_dir=json_dir,
                split=self.split,
                chip_size=self.chip_size,
                patch_size=self.patch_size,
                use_phase=self.use_phase,
                dataset=self.dataset,
            )
