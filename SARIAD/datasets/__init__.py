from .datamodules.image.mstar.mstar import DATASET_INFO as _MSTAR_INFO, MSTAR
from .datamodules.image.hrsid.hrsid import DATASET_INFO as _HRSID_INFO, HRSID
from .datamodules.image.ssdd.ssdd import DATASET_INFO as _SSDD_INFO, SSDD
from .datamodules.image.sample_public.sample_public import DATASET_INFO as _SAMPLE_INFO, SAMPLE_PUBLIC
from .datamodules.image.sardet.sardet import DATASET_INFO as _SARDET_INFO, SARDet_100K

__all__ = ["MSTAR", "HRSID", "SSDD", "SAMPLE_PUBLIC", "SARDet_100K"]  # the datamodules; the runner resolves names from this list

# Documentation metadata of every dataset, keyed by datamodule class name (used to generate the docs).
DATASETS_INFO = {
    "MSTAR": _MSTAR_INFO,
    "HRSID": _HRSID_INFO,
    "SSDD": _SSDD_INFO,
    "SAMPLE_PUBLIC": _SAMPLE_INFO,
    "SARDet_100K": _SARDET_INFO,
}
