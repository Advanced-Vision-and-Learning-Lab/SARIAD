from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from SARIAD.utils.img_utils import img_debug
from SARIAD.config import DEBUG

import torch
import torch.nn.functional as F

class MedianFilter_Transform(Transform):
    def __init__(self, model_transform, kernel_size=3, use_cuda=True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.pre_denoise_transforms = Compose([
            model_transform,
            Grayscale()
        ])

    def _transform(self, inpt: torch.Tensor, params=None):
        original_device = inpt.device
        original_dtype = inpt.dtype
        batch_dim_present = (inpt.dim() == 4)

        if DEBUG:
            # Capture original image for debug BEFORE any transformations are applied
            if batch_dim_present:
                original_image_for_debug = inpt[0].cpu().permute(1, 2, 0).numpy()
            else:
                original_image_for_debug = inpt.cpu().permute(1, 2, 0).numpy()

        if not batch_dim_present:
            inpt = inpt.unsqueeze(0)

        processed_input_list = []
        for img_tensor in inpt:
            img_tensor_float = img_tensor.float() # Explicitly convert to float
            processed_input_list.append(self.pre_denoise_transforms(img_tensor_float.cpu()))
        
        processed_input = torch.stack(processed_input_list)
        processed_input = processed_input.float()

        if self.use_cuda:
            processed_input = processed_input.to(torch.device("cuda"))

        denoised_output_batch = []
        for img in processed_input:
            denoised_output_batch.append(self._apply_median_single_image(img.squeeze(0)))

        final_output = torch.stack(denoised_output_batch)

        if final_output.dim() == 3:
            final_output = final_output.unsqueeze(1)
        
        if final_output.shape[1] == 1:
            final_output = final_output.repeat(1, 3, 1, 1)

        final_output = final_output.to(original_device).to(original_dtype)

        if DEBUG:
            if final_output.shape[1] == 3:
                denoised_image_np = final_output[0].cpu().permute(1, 2, 0).numpy()
            else:
                denoised_image_np = final_output[0].cpu().squeeze(0).numpy()

            img_debug(title=f"Median Filtered Image (Kernel {self.kernel_size})", Original_Input=original_image_for_debug, Denoised_Output=denoised_image_np)

        return final_output

    def _apply_median_single_image(self, img: torch.Tensor):
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] == 1:
            img = img.unsqueeze(0)
        else:
            raise ValueError("Input image must be (H, W) or (1, H, W) for single image median filter.")

        patches = F.unfold(img, kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding)
        patches = patches.view(img.shape[0], img.shape[1], self.kernel_size * self.kernel_size, -1)
        median_values = torch.median(patches, dim=2).values
        denoised_image = median_values.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3])

        return denoised_image.squeeze(0).squeeze(0)


class MedianFilter(PreProcessor):
    def __init__(self, model_transform, kernel_size=3, use_cuda=True):
        super().__init__()

        self.transform = MedianFilter_Transform(model_transform, kernel_size, use_cuda)
        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)
