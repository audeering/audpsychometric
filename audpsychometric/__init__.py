import audpsychometric.core
from audpsychometric.core import datasets
from audpsychometric.core.datasets import list_datasets
from audpsychometric.core.datasets import read_dataset
from audpsychometric.core.gold_standard import evaluator_weighted_estimator
from audpsychometric.core.gold_standard import gold_standard_mean
from audpsychometric.core.gold_standard import gold_standard_median
from audpsychometric.core.gold_standard import gold_standard_mode
from audpsychometric.core.gold_standard import rater_confidence_pearson
import audpsychometric.core.reliability
from audpsychometric.core.reliability import congeneric_reliability
from audpsychometric.core.reliability import cronbachs_alpha
from audpsychometric.core.reliability import intra_class_correlation


# Disencourage from audpsychometric import *
__all__ = []

# Dynamically get the version of the installed module
try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
