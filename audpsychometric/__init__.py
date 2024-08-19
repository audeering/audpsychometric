"""Library to facilitate evaluation and processing of annotated speech.

The input data format for all functions in this module is the same:
a :class:`pd.DataFrame` is expected.

This dataframe is assumed to

- have an index that identifies the unit of observation,
  e.g. a psychometric item to be rated
- have a separate column for each rater

So the entry in the frame at (irow, icol)
identifies the rating of unit irow by rather icol.
Note that these assumptions are not checked
and are under responsibility of the user.

"""
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
