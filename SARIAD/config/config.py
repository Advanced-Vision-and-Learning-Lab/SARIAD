import os

PROJECT_ROOT = os.getcwd()
# Where datasets are downloaded/generated. Override with the SARIAD_DATASETS_PATH environment
# variable; individual datamodules can also be pointed elsewhere with their `path` argument.
DATASETS_PATH = os.environ.get("SARIAD_DATASETS_PATH", os.path.join(PROJECT_ROOT, "datasets"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
