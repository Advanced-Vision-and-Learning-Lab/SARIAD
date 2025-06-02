from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform

import torch
import numpy as np
import pickle
import os

DIR = os.path.dirname(__file__)

# Load the SAR-CNN 2017 network once
try:
    from .SARCNN_SRC.models.DnCNN import DnCNN
    with open(f"{DIR}/SARCNN_SRC/weights/sar_sync/SAR_CNN_e50.pkl", "rb") as fid:
        dncnn_opt = dict(**pickle.load(fid).dncnn)
        dncnn_opt["residual"] = True
    SAR_CNN_NET = DnCNN(1, 1, **dncnn_opt)
    SAR_CNN_NET.load_state_dict(torch.load(f'{DIR}/SARCNN_SRC/weights/sar_sync/SAR_CNN_e50.t7')['net'])
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

class SARCNN_DenoisingTransform(Transform):
    """
    Custom transform to apply SAR-CNN denoising.
    """
    def __init__(self, use_cuda: bool = True, noise_seed: int = 32):
        super().__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.noise_seed = noise_seed
        self.random_stream = np.random.RandomState(self.noise_seed)
        self.net = SAR_CNN_NET
        if self.net is None:
            raise RuntimeError("SAR_CNN_NET was not loaded. Cannot initialize SARCNN_DenoisingTransform.")
        
        if self.use_cuda:
            self.net = self.net.cuda()

    def _transform(self, inpt: torch.Tensor, params=None):
        """
        Applies the SAR denoising process to the input image tensor.
        inpt: A torch.Tensor representing the image (C, H, W).
              Anomalib works with (B, C, H, W), so we'll handle batch dimension.
        """
        # Ensure image is float32 and has a batch dimension if it doesn't already
        # Anomalib's transforms usually receive (C, H, W) for single images,
        # but inside a batch it's (B, C, H, W). Let's assume (B, C, H, W) or (C, H, W)
        
        # Add batch dimension if it's missing (e.g., from dataloader output before collate)
        if inpt.dim() == 3:
            inpt = inpt.unsqueeze(0) # Add batch dimension
        elif inpt.dim() != 4:
            raise ValueError(f"Expected image tensor to have 3 or 4 dimensions (C, H, W) or (B, C, H, W), but got {inpt.dim()}")

        if inpt.shape[1] != 1:
            raise ValueError("SAR_DenoisingTransform expects a single-channel image (C=1).")
        
        original_shape = inpt.shape
        
        if self.use_cuda:
            inpt = inpt.cuda()

        with torch.no_grad():
            processed_input = preprocessing_int2net(inpt)
            denoised_output = self.net(processed_input)
            final_output = postprocessing_net2int(denoised_output)

        # Ensure output is on CPU and matches input dtype if necessary, or just return as is
        return final_output.cpu().reshape(original_shape)

class SARCNN_Denoising(PreProcessor):
    """
    A custom PreProcessor for Anomalib that integrates the SAR_DenoisingTransform.
    """
    def __init__(self, use_cuda: bool = True, noise_seed: int = 32):
        super().__init__()
        self.sar_denoise_transform = SARCNN_DenoisingTransform(use_cuda=use_cuda, noise_seed=noise_seed)
        
        # Anomalib expects exportable transforms for ONNX export etc.
        self.export_transform = get_exportable_transform(self.sar_denoise_transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.sar_denoise_transform(batch.image)
