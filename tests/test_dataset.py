"""Test Dataset Module."""
import pandas as pd
import pytest

import audpsychometric


def test_list_datasets():
    """First basic dataset is available in dataset list."""
    df_datasets = audpsychometric.list_datasets()
    assert "statology" in df_datasets.index


@pytest.mark.parametrize(
    "dataset",
    [
        "statology",
        "hallgren-table5",
        "hallgren-table3",
        "HolzingerSwineford1939",
        "Shrout_Fleiss",
        "wine",
    ],
)
def test_read_dataset(dataset):
    """Test functional requirement that a dataset can be read into dataframe."""
    df_dataset = audpsychometric.read_dataset(dataset)
    assert isinstance(df_dataset, pd.DataFrame)
