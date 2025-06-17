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

        self.pre_transform = Compose([
            Grayscale(),
            model_transform,
        ])

    def transform(self, inpt: torch.Tensor, params=None):
        original_device = inpt.device
        original_dtype = inpt.dtype
        
        if DEBUG:
            original_image = inpt[0].cpu().permute(1, 2, 0).numpy()

        processed_input_list = []
        for img_tensor in inpt:
            img_tensor_float = img_tensor.float()
            processed_input_list.append(self.pre_transform(img_tensor_float.cpu()))
        
        processed_input = torch.stack(processed_input_list)
        
        if self.use_cuda:
            processed_input = processed_input.to(torch.device("cuda"))

        denoised_output_batch = []
        for img_single_channel in processed_input:
            denoised_output_batch.append(self._apply_median_single_image(img_single_channel))

        final_output = torch.stack(denoised_output_batch)

        # if final_output.shape[1] == 1 and original_image.shape[-1] == 3:
        #     final_output = final_output.repeat(1, 3, 1, 1)

        final_output = final_output.to(original_device).to(original_dtype)

        if DEBUG:
            if final_output.shape[1] == 3:
                denoised_image = final_output[0].cpu().permute(1, 2, 0).numpy()
            else:
                denoised_image = final_output[0].cpu().squeeze(0).numpy()

            img_debug(title=f"Median Filtered Image (Kernel {self.kernel_size})", Original_Input=original_image, Denoised_Output=denoised_image)

        return final_output

    def _apply_median_single_image(self, img: torch.Tensor):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input image dimensions for single median filter: {img.shape}. Expected (C, H, W).")

        img = img.float()

        patches = F.unfold(img, kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding)
        patches = patches.view(img.shape[0], img.shape[1], self.kernel_size * self.kernel_size, -1)
        median_values = torch.median(patches, dim=2).values
        denoised_image = median_values.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3])

        return denoised_image.squeeze(0)

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
