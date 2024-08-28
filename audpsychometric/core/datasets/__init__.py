__all__ = ["read_dataset", "list_dataset"]

import os

import pandas as pd


data_directory = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(data_directory, "datasets.csv")
data_sets = pd.read_csv(dataset_path, sep=",")


def read_dataset(data_set_name: str) -> pd.DataFrame:
    r"""Read dataset identified by name.

    retrieves a test dataset from within the package.

    Args:
        data_set_name(str): dataset name

    Returns:
        dataframe containing dataset

    Examples:
        >>> df = read_dataset("wine")
        >>> df.head()
           Wine Judge  Scores
        0     1     A       1
        1     2     A       1
        2     3     A       3
        3     4     A       6
        4     5     A       6

    """
    ds = data_sets.loc[data_sets["dataset"] == data_set_name]

    fname = ds["fname"].values[0]
    fpath = os.path.join(data_directory, fname)
    df = pd.read_csv(fpath, sep=",")
    return df


def list_datasets():
    r"""List tests datasets available in package.

    Returns:
        dataframe listing available datasets

    Examples:
        >>> list_datasets()
                                                     fname  ...                   description
        dataset                                             ...
        statology                            statology.csv  ...      icc sample from web page
        hallgren-table5              Hallgren-Table-05.csv  ...    icc table from publication
        hallgren-table3              Hallgren-Table-03.csv  ...  kappa table from publication
        HolzingerSwineford1939  HolzingerSwineford1939.csv  ...                        lavaan
        Shrout_Fleiss               Shrout_Fleiss_1979.csv  ...            Dataset from paper
        wine                                      wine.csv  ...                 online source
        <BLANKLINE>
        [6 rows x 4 columns]

    """  # noqa: E501
    df_data_sets = data_sets.set_index("dataset")
    return df_data_sets
