from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from torchvision import tv_tensors
from SARIAD.utils.img_utils import img_debug
from SARIAD.config import DEBUG
from typing import Any, Dict, List

import torch
from torch_nlm import nlm2d

class NLM_Transform(Transform):
    def __init__(self, model, std=0.1, kernel_size=21, use_cuda=True):
        super().__init__()
        self.std = std
        self.kernel_size = kernel_size
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

            denoised_output_batch = []
            for img in processed_input:
                if img.dim() == 3:
                    img_grayscale = img.mean(dim=0, keepdim=False)
                    denoised_single_channel = nlm2d(img_grayscale, kernel_size=self.kernel_size, std=self.std)
                    denoised_output_batch.append(denoised_single_channel.unsqueeze(0).repeat(3, 1, 1))
                else:
                    raise ValueError("Shape for NLM must be 3D (CxHxW)")

            final_output = torch.stack(denoised_output_batch)
            final_output = final_output.to(original_device).to(original_dtype)
            return final_output
        return self.pre_transform(inpt)

class NLM(PreProcessor):
    def __init__(self, model, std=0.1, kernel_size=21, use_cuda=True):
        super().__init__()

        self.transform = NLM_Transform(model, std, kernel_size, use_cuda)
        # self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
