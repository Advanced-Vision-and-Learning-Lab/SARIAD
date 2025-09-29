import logging
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from anomalib.data import Batch
from anomalib.models.components import AnomalibModule
from anomalib import LearningType
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit_mae import HiViTMaskedAutoencoder
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit import HiViT
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.pos_embed import interpolate_pos_embed
from SARIAD.utils.blob_utils import fetch_blob
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Grayscale

logger = logging.getLogger(__name__)

class SARATRX(AnomalibModule):
    def __init__(self, pre_processor=True, post_processor=True, num_classes=2):
        transform = Compose([
            Resize((224, 224)),
            Grayscale(num_output_channels=1),
        ])
        super().__init__(PreProcessor(transform), post_processor)

        self.trainer_arguments = {
            "max_epochs": 100,
            "accelerator": "gpu",
            "devices": 1,
            "check_val_every_n_epoch": 1,
            "callbacks": [],
            "logger": True,
        }

        self.model = HiViTMaskedAutoencoder(hifeat=True)
        print(self.model)
        self.outputs = []

        fetch_blob("mae_hivit_base_1600ep.pth", drive_file_id="1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ", is_archive=False);
        checkpoint = torch.load("mae_hivit_base_1600ep.pth", map_location='cpu')
        # print(checkpoint)

        # checkpoint_model = checkpoint
        # state_dict = self.model.state_dict()
        # print(len(state_dict.keys()))
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        # interpolate_pos_embed(self.model, checkpoint_model)

        msg = self.model.load_state_dict(checkpoint, strict=False)
        model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

    def config_pre_processor(self):
        return PostProcessor()

    def configure_post_processor(self):
        return PostProcessor()

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        del args, kwargs  # These variables are not used.

        _ = self.model(batch.image)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def configure_optimizers(self):
        pass

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def learning_type(self):
        return LearningType.ONE_CLASS

    def trainer_arguments(self):
        pass

if __name__ == "__main__":
    SARATRX()
