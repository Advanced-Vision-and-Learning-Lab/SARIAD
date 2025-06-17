from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from SARIAD.utils.img_utils import img_debug
from SARIAD.config import DEBUG

import torch
from torch_nlm import nlm2d

class NLM_Transform(Transform):
    def __init__(self, model_transform, std=0.1, kernel_size=21, use_cuda=True):
        super().__init__()
        self.std = std
        self.kernel_size = kernel_size
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.pre_transform = Compose([
            model_transform,
            Grayscale()
        ])

    def transform(self, inpt: torch.Tensor, params=None):
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
                denoised_output_batch.append(nlm2d(img.squeeze(0), kernel_size=self.kernel_size, std=self.std))
            else:
                raise ValueError("Shape for NLM must be 2D (CxHxW)")

        final_output = torch.stack(denoised_output_batch)

        if final_output.dim() == 3:
            final_output = final_output.unsqueeze(1)
        
        if final_output.shape[1] == 1:
            final_output = final_output.repeat(1, 3, 1, 1)

        final_output = final_output.to(original_device).to(original_dtype)

        if DEBUG:
            if inpt.shape[1] == 3:
                original_image_np = inpt[0].cpu().permute(1, 2, 0).numpy()
            else:
                original_image_np = inpt[0].cpu().squeeze(0).numpy()

            if final_output.shape[1] == 3:
                denoised_image_np = final_output[0].cpu().permute(1, 2, 0).numpy()
            else:
                denoised_image_np = final_output[0].cpu().squeeze(0).numpy()

            img_debug(title="NLM Denoised Image (First in Batch)", Original_Input=original_image_np, Denoised_Output=denoised_image_np)

        return final_output

class NLM(PreProcessor):
    def __init__(self, model_transform, std=0.1, kernel_size=21, use_cuda=True):
        super().__init__()

        self.transform = NLM_Transform(model_transform, std, kernel_size, use_cuda)
        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)
