from audpsychometric.core.datasets import list_datasets
from audpsychometric.core.datasets import read_dataset
from audpsychometric.core.gold_standard import agreement_categorical
from audpsychometric.core.gold_standard import agreement_numerical
from audpsychometric.core.gold_standard import evaluator_weighted_estimator
from audpsychometric.core.gold_standard import mode
from audpsychometric.core.gold_standard import rater_agreement
from audpsychometric.core.reliability import congeneric_reliability
from audpsychometric.core.reliability import cronbachs_alpha
from audpsychometric.core.reliability import intra_class_correlation


# Disencourage from audpsychometric import *
__all__ = []

# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
