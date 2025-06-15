from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from SARIAD.utils.img_utils import img_debug

import torch
import torch.nn.functional as F

class NLM_Transform(Transform):
    def __init__(self, model_transform, h=0.1, patch_size=7, search_window_size=21, use_cuda=True, debug=False):
        super().__init__()
        self.h = h
        self.patch_size = patch_size
        self.search_window_size = search_window_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.debug = debug

        if self.patch_size % 2 == 0 or self.search_window_size % 2 == 0:
            raise ValueError("patch_size and search_window_size must be odd.")

        self.patch_half_size = self.patch_size // 2
        self.search_half_size = self.search_window_size // 2

        self.pre_denoise_transforms = Compose([
            model_transform,
            Grayscale()
        ])

    def _transform(self, inpt: torch.Tensor, params=None):
        original_device = inpt.device
        original_dtype = inpt.dtype
        batch_dim_present = (inpt.dim() == 4)

        if not batch_dim_present:
            inpt = inpt.unsqueeze(0)

        processed_input_list = [self.pre_denoise_transforms(img_tensor.cpu()) for img_tensor in inpt]
        processed_input = torch.stack(processed_input_list)

        if self.use_cuda:
            processed_input = processed_input.to(torch.device("cuda"))

        processed_input = processed_input.float()

        denoised_output_batch = []
        for img in processed_input:
            denoised_output_batch.append(self._apply_nlm_single_image(img.squeeze(0)))

        final_output = torch.stack(denoised_output_batch)

        if final_output.dim() == 3:
            final_output = final_output.unsqueeze(1)
        
        if final_output.shape[1] == 1:
            final_output = final_output.repeat(1, 3, 1, 1)

        final_output = final_output.to(original_device).to(original_dtype)

        if self.debug:
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

    def _apply_nlm_single_image(self, img: torch.Tensor):
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] == 1:
            img = img.unsqueeze(0)
        else:
            raise ValueError("Input image must be (H, W) or (1, H, W) for single image NLM.")

        padded_img = F.pad(img, (self.search_half_size, self.search_half_size,
                                  self.search_half_size, self.search_half_size), mode='reflect')

        H, W = img.shape[-2:]
        denoised_image = torch.zeros_like(img)

        patches = padded_img.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)

        for i in range(H):
            for j in range(W):
                p_patch = patches[:, :, i, j, :, :].reshape(1, -1)

                search_start_i = max(0, i - self.search_half_size)
                search_end_i = min(H, i + self.search_half_size + 1)
                search_start_j = max(0, j - self.search_half_size)
                search_end_j = min(W, j + self.search_half_size + 1)

                search_region_patches = patches[:, :,
                                                search_start_i : search_end_i,
                                                search_start_j : search_end_j,
                                                :, :]
                search_region_patches_flat = search_region_patches.reshape(1, -1, self.patch_size * self.patch_size)

                if search_region_patches_flat.numel() == 0:
                    continue

                diff = search_region_patches_flat - p_patch.unsqueeze(1)
                squared_diff = torch.sum(diff**2, dim=-1)

                weights = torch.exp(-squared_diff / (self.h**2))
                
                search_grid_i, search_grid_j = torch.meshgrid(
                    torch.arange(search_start_i, search_end_i, device=img.device),
                    torch.arange(search_start_j, search_end_j, device=img.device),
                    indexing='ij'
                )
                
                iq_values_flat = img[0, 0, search_grid_i.flatten(), search_grid_j.flatten()]

                sum_weights = torch.sum(weights)
                if sum_weights > 0:
                    normalized_weights = weights / sum_weights
                else:
                    normalized_weights = torch.zeros_like(weights)

                denoised_pixel = torch.sum(normalized_weights * iq_values_flat.float().reshape(normalized_weights.shape))
                denoised_image[0, 0, i, j] = denoised_pixel
        
        return denoised_image.squeeze(0).squeeze(0)


class NLM(PreProcessor):
    def __init__(self, model_transform, h=0.1, patch_size=7, search_window_size=21, use_cuda=True, debug=False):
        super().__init__()
        self.transform = NLM_Transform(model_transform, h, patch_size, search_window_size, use_cuda, debug)

        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)
