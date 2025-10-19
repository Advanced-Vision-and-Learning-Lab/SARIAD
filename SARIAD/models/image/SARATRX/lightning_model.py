import logging
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from anomalib.data import Batch
from anomalib.models.components import AnomalibModule
from anomalib import LearningType
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit_mae import HiViTMaskedAutoencoder
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit import HiViT
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.lr_decay import param_groups_lrd
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.misc import NativeScalerWithGradNormCount as NativeScaler
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
            "accelerator": "gpu",
            "devices": 1,
            "check_val_every_n_epoch": 1,
            "callbacks": [],
            "logger": True,
        }
        self.automatic_optimization = False

        self.model = HiViTMaskedAutoencoder(hifeat=True)
        print(self.model)

        fetch_blob("mae_hivit_base_1600ep.pth", drive_file_id="1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ", is_archive=False);
        checkpoint = torch.load("mae_hivit_base_1600ep.pth", map_location='cpu')
        print(checkpoint)

        checkpoint_model = checkpoint
        state_dict = self.model.state_dict()
        print(len(state_dict.keys()))
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(self.model, checkpoint_model)

        msg = self.model.load_state_dict(checkpoint, strict=False)
        model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = 64
        
        blr = 1e-3
        lr = blr * eff_batch_size / 256

        print("base lr: %.2e" % (lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % lr)

        print("accumulate grad iterations: %d" % 1)
        print("effective batch size: %d" % eff_batch_size)

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = param_groups_lrd(model_without_ddp, 1,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=0.75
        )
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        self.loss_scaler = NativeScaler()

    def config_pre_processor(self):
        return PostProcessor()

    def configure_post_processor(self):
        return PostProcessor()

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """
        Implements the MAE training step with NativeScaler and manual optimization.
        """
        
        optimizer = self.optimizer
        optimizer.zero_grad() 

        with torch.cuda.amp.autocast():
            loss, _, _ = self.model(batch.image)  
            
        self.loss_scaler(
            loss, 
            optimizer, 
            clip_grad=None, 
            parameters=self.model.parameters(),
        )

        self.log("train_mae_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def configure_optimizers(self):
        """Returns None to signal PyTorch Lightning to use Manual Optimization."""
        return None

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        # _ = self.model(batch.image)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def learning_type(self):
        return LearningType.ONE_CLASS

    def trainer_arguments(self):
        pass

def unpatch(x, patch_size=16):
    h = w = 14
    chans = 3

    #reshape flat patches into 2D image
    x = x.reshape(B, h, w, patch_size, patch_size, chans)

    #shape closer to pytorch (B, C, H, W)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, patch_size, H, patch_size, W)

    x = x.reshape(B, chans, h*patch_size, w*patch_size)
    return x

if __name__ == "__main__":
    import os
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    if __name__ == "__main__":
        # Instantiate model
        model = SARATRX()
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        image_dir = "imgs/"
        result_dir = "results/"
        os.makedirs(result_dir, exist_ok=True)

        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Found {len(image_files)} images in {image_dir}")

        for img_path in image_files:
            print(f"Processing: {img_path}")

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                loss, pred, mask = model.model(tensor)

            print(f"Loss: {loss.item():.4f}")

            # reshape mask
            B = mask.shape[0]
            num_patches = mask.shape[1]
            h = w = int(num_patches ** 0.5)
            mask = mask.reshape(B, 1, h, w)
            mask = F.interpolate(mask, size=(224, 224), mode="nearest")

            # Prepare tensors
            input_img = tensor.squeeze().cpu()
            print(pred.shape)
            recon_img = pred.squeeze().cpu()
            mask = mask.squeeze().cpu()

            def normalize(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)

            input_img = normalize(input_img)
            recon_img = unpatch(pred.cpu())
            recon_img = recon_img.squeeze(0).permute(1, 2, 0)
            masked_input = input_img * (1 - mask)

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(input_img, cmap="gray")
            axes[0].set_title("Input")
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Mask")
            axes[2].imshow(recon_img, cmap="gray")
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"{os.path.basename(img_path)} | Loss={loss.item():.4f}")
            plt.tight_layout()

            out_path = os.path.join(result_dir, os.path.basename(img_path).replace(".png", "_recon.png"))
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")
