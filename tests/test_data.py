"""Test data module (unit tests).

For simplicity, test cases are not documented through docstrings. Yet,
they should be semantically very clear.
"""
import pytest
import pandas as pd
import numpy as np

from context import diamond         # NOQA
from diamond import data

DATASET_LOCATION = 'datasets/diamonds/diamonds.csv'


@pytest.fixture
def raw_filename() -> str:
    """Pytest fixture: return path to dataset.
    """
    return DATASET_LOCATION


@pytest.fixture
def raw_dataframe(raw_filename) -> pd.DataFrame:
    """Pytest fixture: load unprocessed dataframe."""
    return data.load_raw(raw_filename)


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


class TestVolumeFeatureExtractor:

    @pytest.mark.parametrize('input_name, expected', (('x', 'x'),
                                                      (None, 'volume')))
    def test_init(self, input_name, expected):
        if input_name is None:
            extractor = data.VolumeFeatureExtractor()
        else:
            extractor = data.VolumeFeatureExtractor(expected)

        assert extractor.extracted_feature_name == expected

    def test_extract(self, raw_dataframe):
        extractor = data.VolumeFeatureExtractor()
        extracted = extractor.extract(raw_dataframe)
        assert (extracted >= raw_dataframe.x).all()
        assert (extracted >= raw_dataframe.y).all()
        assert (extracted >= raw_dataframe.z).all()
