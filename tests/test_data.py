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


def test_color_encoder():
    color_column = data.color_encoder.fit_transform(
        np.array(data.COLORS).reshape(-1, 1))

    assert (color_column[:, 0] == range(23)).all()


def test_clarity_encoder():
    color_column = data.clarity_encoder.fit_transform(
        np.array(data.CLARITIES).reshape(-1, 1))

    assert (color_column[:, 0] == range(11)).all()


class TestFeatureExtractor:

    def test_abc(self):
        with pytest.raises(TypeError):
            data._FeatureExtractor('...')
