from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform, Compose
import torch

class Default_Transform(Transform):
    def __init__(self, model_transform):
        super().__init__()
        print(model_transform)
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

# from anomalib.engine import Engine
# from anomalib.models import Padim
# from anomalib.data import MVTecAD

# datamodule = MVTecAD(
#     category="bottle",  # MVTec category to use
#     train_batch_size=32,  # Number of images per training batch
#     eval_batch_size=32,  # Number of images per validation/test batch
#     num_workers=8,  # Number of parallel processes for data loading
# )
# datamodule.prepare_data()
# datamodule.setup()

# i, train_data = next(enumerate(datamodule.train_dataloader()))
# print("Batch Image Shape", train_data.image.shape)
# model = Padim(pre_processor=Default(model_transform = Padim.configure_pre_processor().transform))

# engine = Engine()
# engine.fit(model=model, datamodule=datamodule)
# test_results = engine.test(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
# )
