import SARIAD
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib import TaskType
from anomalib.deploy import ExportType

# load a SAR datamodules
from SARIAD.datasets import MSTAR
datamodule = MSTAR()
# from SARIAD.datasets import HRSID
# datamodule = HRSID()
# from SARIAD.datasets import SSDD
# datamodule = SSDD()
datamodule.setup()

i, train_data = next(enumerate(datamodule.train_dataloader()))
print("Batch Image Shape", train_data.image.shape)

# load the PaDiM model
model = Padim()

# load a SAR pre processors
# from SARIAD.pre_processing import SARCNN
# model = Padim(pre_processor=SARCNN())
# from SARIAD.pre_processing import NLM
# model = Padim(pre_processor=NLM())
# from SARIAD.pre_processing import MedianFilter
# model = Padim(pre_processor=MedianFilter())

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# test model
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
