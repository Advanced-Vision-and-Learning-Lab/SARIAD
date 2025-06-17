import os

PROJECT_ROOT = os.getcwd()
DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
