from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose, Grayscale
from SARIAD.utils.img_utils import *
from SARIAD.config import DEBUG

import torch
import numpy as np
import pickle
import os
from pathlib import Path

DIR = os.path.dirname(__file__)

# Load the SAR-CNN 2017 network once
try:
    from .SARCNN_SRC.models.DnCNN import DnCNN
    with open(Path(f"{DIR}/SARCNN_SRC/weights/sar_sync/SAR_CNN_e50.pkl"), "rb") as fid:
        dncnn_opt = dict(**pickle.load(fid).dncnn)
        dncnn_opt["residual"] = True
    SAR_CNN_NET = DnCNN(1, 1, **dncnn_opt)
    SAR_CNN_NET.load_state_dict(torch.load(Path(f"{DIR}/SARCNN_SRC/weights/sar_sync/SAR_CNN_e50.t7"))['net'])
    SAR_CNN_NET.eval()
    if torch.cuda.is_available():
        SAR_CNN_NET = SAR_CNN_NET.cuda()

except FileNotFoundError as e:
    print(f"Error loading SAR-CNN model or weights: {e}")
    print("Please ensure 'models/DnCNN.py' and 'weights/sar_sync/' directory exist and contain the necessary files.")
    SAR_CNN_NET = None

def preprocessing_int2net(img):
    """
    Transforms the image from intensity domain to network input domain.
    Assumes img is a torch.Tensor.
    """
    return img.abs().log() / 2

def postprocessing_net2int(img):
    """
    Transforms the network output back to the intensity domain.
    Assumes img is a torch.Tensor.
    """
    return (2 * img).exp()

class SARCNN_Transform(Transform):
    """
    Custom transform to apply SAR-CNN denoising, grayscale conversion,
    and then convert back to 3 channels.
    """
    def __init__(self, model, use_cuda, noise_seed = 32):
        super().__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.noise_seed = noise_seed
        self.random_stream = np.random.RandomState(self.noise_seed)
        self.net = SAR_CNN_NET
        if self.net is None:
            raise RuntimeError("SAR_CNN_NET was not loaded. Cannot initialize SARCNN_Transform.")
        
        if self.use_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        self.pre_transform = Compose([
            *model.configure_pre_processor().transform.transforms, 
        ])

    def transform(self, inpt: torch.Tensor, params=None):
        """
        Applies the SAR denoising process to the input image tensor after initial transforms.
        inpt: A torch.Tensor representing the image (C, H, W) or (B, C, H, W).
        """
        original_device = inpt.device
        original_dtype = inpt.dtype
        batch_dim_present = (inpt.dim() == 4)
        if batch_dim_present:
            original_image = inpt[0].cpu().permute(1, 2, 0).numpy()
        else:
            original_image = inpt.cpu().permute(1, 2, 0).numpy()

        if batch_dim_present:
            processed_input_list = [self.pre_transform(img_tensor.cpu()) for img_tensor in inpt]
            processed_input = torch.stack(processed_input_list)
        else:
            processed_input = self.pre_transform(inpt.cpu())
            processed_input = processed_input.unsqueeze(0)

        # Move to GPU if self.use_cuda is true, otherwise keep on CPU
        if self.use_cuda:
            processed_input = processed_input.to(torch.device("cuda"))

        with torch.no_grad():
            denoise_input = preprocessing_int2net(processed_input)
            denoised_output = self.net(denoise_input)
            final_output = postprocessing_net2int(denoised_output)

        # Convert back to 3 channels by replicating the single channel
        if final_output.shape[1] == 1:
            final_output = final_output.repeat(1, 3, 1, 1) # replicate channel for B,C,H,W
        else:
            print(f"Warning: DnCNN output has {final_output.shape[1]} channels, not 1. Skipping 3-channel conversion.")

        if DEBUG:
            if batch_dim_present:
                img_debug(title="Final Denoised Image (First in Batch)", Original_Input=original_image,
                          Denoised_Output=final_output[0].cpu().permute(1, 2, 0).numpy())
            else:
                img_debug(title="Final Denoised Image", Original_Input=original_image,
                          Denoised_Output=final_output.cpu().permute(1, 2, 0).numpy())

        return final_output.to(original_device).to(original_dtype)

class SARCNN(PreProcessor):
    """
    A custom PreProcessor for Anomalib that integrates the SAR_DenoisingTransform.
    """
    def __init__(self, model_transform, use_cuda = True, noise_seed = 32):
        super().__init__()
        self.sar_denoise_transform = SARCNN_Transform(
            model_transform,
            use_cuda = use_cuda,
            noise_seed = noise_seed,
        )
        
        self.export_transform = get_exportable_transform(self.sar_denoise_transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)
