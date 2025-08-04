from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from SARIAD.utils.img_utils import img_debug
from SARIAD.config import DEBUG
from torchvision import tv_tensors
from SARIAD.utils.img_utils import img_debug
from SARIAD.config import DEBUG
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

class MedianFilter_Transform(Transform):
    def __init__(self, model, kernel_size=3, use_cuda=True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.pre_transform = model.configure_pre_processor().transform

    def transform(self, inpt: Any, params: Dict[str, Any]):
        if type(inpt) == tv_tensors._image.Image:
            original_device = inpt.device
            original_dtype = inpt.dtype

            processed_input_list = [self.pre_transform(img_tensor.cpu()) for img_tensor in inpt]
            processed_input = torch.stack(processed_input_list)

            if self.use_cuda:
                processed_input = processed_input.to(torch.device("cuda"))

            processed_input = processed_input.float()

            final_output = self._apply_median_batch(processed_input)

            final_output = final_output.to(original_device).to(original_dtype)

            return final_output
        return self.pre_transform(inpt)

    def _apply_median_batch(self, batch_img: torch.Tensor):
        batch_img = batch_img.float()
        
        patches = F.unfold(batch_img, kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding)
        patches = patches.view(batch_img.shape[0], batch_img.shape[1], self.kernel_size * self.kernel_size, -1)
        median_values = torch.median(patches, dim=2).values
        denoised_image = median_values.view(batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3])

        return denoised_image

class MedianFilter(PreProcessor):
    def __init__(self, model, kernel_size=3, use_cuda=True):
        super().__init__()

        self.transform = MedianFilter_Transform(model, kernel_size, use_cuda)
        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
