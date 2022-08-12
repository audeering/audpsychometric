"""Provide example datasets for package
"""


__all__ = ["read_dataset", "list_dataset"]

import os

import pandas as pd

data_directory = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(data_directory, 'datasets.csv')
data_sets = pd.read_csv(dataset_path, sep=',')


def read_dataset(data_set_name: str) -> pd.DataFrame:
    r"""read dataset identified by name.

    retrieves a test dataset from within the package.

    Args:
        data_set_name(str): string identifier of the dataset.
        This does not need not be identical with the filename

    Returns:
        table containing dataset


    """

    ds = data_sets.loc[data_sets["dataset"] == data_set_name]

    fname = ds['fname'].values[0]
    fpath = os.path.join(data_directory, fname)
    df = pd.read_csv(fpath, sep=',')
    return df


def list_datasets():
    r'''List tests datasets available in package

    Args:
        None
    Returns:
        table listing available datasets

    '''

    df_data_sets = data_sets.set_index('dataset')
    return df_data_sets
