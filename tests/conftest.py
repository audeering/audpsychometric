import numpy as np
import pytest

import audpsychometric


@pytest.fixture(scope="function")
def df_holzinger_swineford():
    df_dataset = audpsychometric.read_dataset("HolzingerSwineford1939")
    cols_use = [col for col in df_dataset.columns if col.startswith("x")]
    df = df_dataset[cols_use].astype(np.float32)
    return df


@pytest.fixture(scope="session", autouse=True)
def fixture_session():
    # always run this once when a session is started
    print("\nenter test session")
    yield
    # always run this once when a session has ended
    print("\nleave test session")


@pytest.fixture(scope="function")
def fixture_function():
    # run this before every test with this decorator:
    # @pytest.mark.usefixtures('fixture_function')
    print("\nenter test function")
    yield
    # run this after every test with this decorator:
    # @pytest.mark.usefixtures('fixture_function')
    print("\nleave test function")
