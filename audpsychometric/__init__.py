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
import audpsychometric.core.reliability
from audpsychometric.core import datasets
from audpsychometric.core.gold_standard import (
    evaluator_weighted_estimator,
    rater_confidence_pearson,
    gold_standard_mean,
    gold_standard_median,
    gold_standard_mode,
)
from audpsychometric.core.reliability import (
    congeneric_reliability,
    cronbachs_alpha,
    intra_class_correlation
)

from audpsychometric.core.datasets import (
    list_datasets,
    read_dataset)

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
