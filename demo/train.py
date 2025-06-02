# Import torch and set matrix multiplication precision
import torch
torch.set_float32_matmul_precision('medium')

from anomalib.engine import Engine
#1 shots
from anomalib.models import Padim
from anomalib import TaskType
from anomalib.deploy import ExportType

# import our SAR datasets
from SARIAD.datasets import MSTAR
from SARIAD.datasets import HRSID
from SARIAD.datasets import SSDD

# import the SAR CNN preprocessor
from SARIAD.pre_processing import SARCNN_Denoising

# load our MSTAR model
datamodule = MSTAR()

# load our HRSID model
# datamodule = HRSID()

datamodule.setup()

i, train_data = next(enumerate(datamodule.train_dataloader()))
print("Batch Image Shape", train_data.image.shape)

# load PaDiM
model = Padim()

# load PaDiM with the SARCNN_Denoising pre_processor
# model = Padim(pre_processor=SARCNN_Denoising())

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# test Model
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)

# export model to for OpenVINO inference
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    datamodule=datamodule,
    export_root="./weights/openvino",
)
