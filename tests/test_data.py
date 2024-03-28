"""Test data module (unit tests).

For simplicity, test cases are not documented through docstrings. Yet,
they should be semantically very clear.
"""
import pytest
import pandas as pd
import numpy as np

from context import diamond         # NOQA
from diamond import data


@pytest.fixture
def raw_filename() -> str:
    """Pytest fixture: return path to dataset.
    """
    return 'datasets/diamonds/diamonds.csv'


def test_load_raw(raw_filename):
    df = data.load_raw(raw_filename)
    assert (df.count() > 0).all()


def test_cut_grades_encoder():
    cut_column = data.cut_grades_encoder.fit_transform(
        np.array(data.CUT_GRADES).reshape(-1, 1))

    assert (cut_column[:, 0] == range(6)).all()