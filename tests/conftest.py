import gc
import warnings

import pytest


@pytest.fixture(autouse=True)
def _cleanup_connections():
    yield
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()
