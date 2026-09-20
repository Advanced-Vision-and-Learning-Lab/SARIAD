from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Transform, Compose
import torch

class Default_Transform(Transform):
    def __init__(self, model_transform):
        super().__init__()
        self.pre_transform = Compose([
            model_transform
        ])

    def transform(self, inpt: torch.tensor, params=None):
        original_device = inpt.device
        original_dtype = inpt.dtype

        processed_inputs = [self.pre_transform(img) for img in inpt]
        processed_inputs = torch.stack(processed_inputs)
        
        return processed_inputs.to(original_device).to(original_dtype)

class Default(PreProcessor):
    def __init__(self, model_transform):
        super().__init__()
        self.transform = Default_Transform(model_transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch.image = self.transform(batch.image)
