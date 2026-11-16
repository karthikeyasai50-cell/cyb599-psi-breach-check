import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psi.data import make_dataset          # noqa: E402
from psi.params import B_PARAMS            # noqa: E402
from psi.protocol_b import BFVServer       # noqa: E402

N_TEST = 5000


@pytest.fixture(scope="session")
def dataset():
    return make_dataset(N_TEST, 100, seed=7)


@pytest.fixture(scope="session")
def bfv_servers(dataset):
    # Built once: server setup and flood calibration take a few seconds each.
    return {name: BFVServer(p, dataset.server, seed=1) for name, p in B_PARAMS.items()}
