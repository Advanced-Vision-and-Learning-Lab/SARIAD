"""SAR-CNN despeckling preprocessor (Chierchia et al., 2017).

The network works in the log domain: it takes ``log(|intensity|) / 2`` and predicts the same
quantity for the clean image, so the prediction is mapped back with ``exp(2 * y)``. That is done
on the *raw* image, before the anomaly model's own resize/normalization is applied.

The network code and weights come from the ``SARCNN_SRC`` git submodule
(``git submodule update --init``); they are loaded lazily the first time the preprocessor is used,
so importing SARIAD never requires the submodule (or a GPU).
"""

import functools
import pickle
from pathlib import Path
from typing import Any, Dict

import torch
from anomalib.pre_processing import PreProcessor
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform

SRC_DIR = Path(__file__).parent / "SARCNN_SRC"
WEIGHTS_DIR = SRC_DIR / "weights" / "sar_sync"

# Smallest intensity fed to the log (a pixel value of exactly 0 would give -inf)
EPS = 1e-6


@functools.lru_cache(maxsize=1)
def load_sarcnn_net() -> torch.nn.Module:
    """Load the pretrained SAR-CNN 2017 network on the CPU (cached)."""
    try:
        from .SARCNN_SRC.models.DnCNN import DnCNN

        with open(WEIGHTS_DIR / "SAR_CNN_e50.pkl", "rb") as fid:
            dncnn_opt = dict(**pickle.load(fid).dncnn)
        dncnn_opt["residual"] = True
        net = DnCNN(1, 1, **dncnn_opt)
        net.load_state_dict(torch.load(WEIGHTS_DIR / "SAR_CNN_e50.t7", map_location="cpu")["net"])
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(
            "Could not load the SAR-CNN network or weights. Make sure the SAR-CNN submodule is "
            f"checked out: `git submodule update --init` (looked in {SRC_DIR})."
        ) from e
    return net.eval()


def preprocessing_int2net(img: torch.Tensor) -> torch.Tensor:
    """Intensity domain -> network input domain."""
    return img.abs().clamp_min(EPS).log() / 2


def postprocessing_net2int(img: torch.Tensor) -> torch.Tensor:
    """Network output domain -> intensity domain."""
    return (2 * img).exp()


class SARCNN_Transform(Transform):
    """Despeckle images with SAR-CNN, then apply the anomaly model's own transform.

    Images are converted to grayscale (the network is single channel), denoised, and repeated
    to three channels like the other SARIAD preprocessors.
    """

    def __init__(self, model, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self._net = None
        self.pre_transform = model.configure_pre_processor().transform

    @property
    def net(self) -> torch.nn.Module:
        if self._net is None:
            self._net = load_sarcnn_net().to(self.device)
        return self._net

    def transform(self, inpt: Any, params: Dict[str, Any]):
        if isinstance(inpt, tv_tensors.Image):
            original_device, original_dtype = inpt.device, inpt.dtype

            gray = inpt.to(self.device).float().mean(dim=1, keepdim=True)
            with torch.no_grad():
                denoised = postprocessing_net2int(self.net(preprocessing_int2net(gray)))
            denoised = denoised.repeat(1, 3, 1, 1).to(original_device).to(original_dtype)

            return torch.stack([self.pre_transform(img.cpu()) for img in denoised]).to(original_device)
        return self.pre_transform(inpt)


class SARCNN(PreProcessor):
    """Anomalib ``PreProcessor`` that despeckles every batch with SAR-CNN."""

    def __init__(self, model, use_cuda=True):
        super().__init__()
        self.transform = SARCNN_Transform(model, use_cuda=use_cuda)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
