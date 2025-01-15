# Import torch and set matrix multiplication precision
import torch
torch.set_float32_matmul_precision('medium')

from anomalib.engine import Engine
#1 shots
from anomalib.models import Padim
from anomalib import TaskType
from anomalib.deploy import ExportType

from mstar import MSTAR

# load our MSTAR model
datamodule = MSTAR()
datamodule.setup()

i, data = next(enumerate(datamodule.val_dataloader()))
images = data['image']
print("Batch image shape:", images.shape)
print(data.keys())

model = Padim()
engine = Engine(task=TaskType.CLASSIFICATION, max_epochs=100)
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
    input_size=(256, 256),
    export_root="./weights/openvino",
)
