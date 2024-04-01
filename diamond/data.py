"""Data pipeline components."""
import abc
from string import ascii_uppercase
from functools import reduce
from operator import and_

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

CUT_GRADES = 'Poor', 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'
COLORS = tuple(reversed(ascii_uppercase[3:]))
CLARITIES = ('I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1',
             'IF', 'FL')

CLEAN_RANGES = {
    'x': (0.1, np.inf, 'left'),
    'y': (0.1, np.inf, 'left'),
    'z': (0.1, np.inf, 'left'),
    'price': (0., np.inf, 'neither')
}
"""Default acceptance ranges for the input data.

To be used with :func:`clean`.
"""


def load_raw(file_or_path) -> pd.DataFrame:
    """Load raw dataset from file (csv only, so far)."""
    return pd.read_csv(file_or_path)


cut_grades_encoder = OrdinalEncoder(categories=[list(CUT_GRADES)],
                                    dtype=np.float32)

color_encoder = OrdinalEncoder(categories=[list(COLORS)],
                               dtype=np.float32)

clarity_encoder = OrdinalEncoder(categories=[list(CLARITIES)],
                                 dtype=np.float32)


def clean(dataset: pd.DataFrame, ranges_dict=CLEAN_RANGES,
          **ranges) -> pd.DataFrame:
    """Clean dataset filtering invalid datapoints, based on ranges.

    ``ranges_dict`` is a dictionary of ranges in the form
    ``{feature_name: (min, max, inclusive), ...}``, defaulting to
    :attr:`CLEAN_RANGES`. Replace it with a different dict or fine tune
    it with keyword arguments. Keyword arguments will be added to the
    ``ranges_dict``, eventually overwriting its values.

    For the meaning of the ``inclusive`` keyworkd in the range tuple,
    see pandas ``pandas.Series.between``.
    """
    updated_ranges = ranges_dict | ranges

    if not updated_ranges:
        return dataset

    return dataset[
        reduce(and_, map(lambda n: dataset[n].between(*updated_ranges[n]),
                         updated_ranges))
    ]


def split(*datasets: pd.DataFrame, holdout: float = 0.2,
          random_state=1245324558) -> list[pd.DataFrame]:
    """Split dataset.

    ``holdout`` is a float in the range ``(0, 1)`` representing the
    proportion of the holdout (test set) with respect to the entire
    dataset. Defaults to 20%.

    In practice, it is just a wrapper function on sklearn's
    ``train_test_split``, but with defaults tailored for the project.
    """
    return train_test_split(*datasets, test_size=holdout,
                            random_state=random_state)


def get_X_y(dataset: pd.DataFrame,
            y_column='price') -> tuple[pd.DataFrame, pd.Series]:
    """Return a dataset split in its features matrix and labels vector."""
    return dataset.drop(columns=y_column), dataset[y_column]


class _FeatureExtractor(BaseEstimator, TransformerMixin, abc.ABC):
    """Base class for a feature extractor transformer.

    Subclass to define a custom feature extraction algorithm
    (see :meth:`extract`). This transformer works with pandas
    ``DataFrame`` instances.
    """

    def __init__(self, extracted_feature_name: str):
        self.extracted_feature_name = extracted_feature_name

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the extractor."""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """Add the extracted feature."""
        X[self.extracted_feature_name] = self.extract(X)
        return X

    @abc.abstractmethod
    def extract(self, X: pd.DataFrame) -> pd.Series:
        """Override to specify the feature extraction algorithm.

        A ``pd.Series`` is the expected output, which will be appended
        to ``X`` by the :meth:`transform` implementation.
        """

    def set_output(self, *, transform):
        """Set output format, in practice only pandas is supported."""
        if transform != 'pandas':
            raise ValueError('This feature extractor only supports pandas '
                             'output format')
        return self


class VolumeFeatureExtractor(_FeatureExtractor):
    r"""Transformer, extract volume feature of diamonds.

    ``x``, ``y`` and ``z`` features are required. Formally the feature
    is defined as:

    :math:`v = x \cdot y \cdot z`.
    """

    def __init__(self, extracted_feature_name: str = 'volume'):
        super().__init__(extracted_feature_name)

    def extract(self, X: pd.DataFrame) -> pd.Series:
        """Extract and return the volume feature."""
        return X.x * X.y * X.z


class EccentricityFeatureExtractor(_FeatureExtractor):
    r"""Transformer, extract eccentricity feature of diamonds.

    ``x``, ``y`` features are required. Formally the feature is defined
    as:

    :math:`e = \sqrt{1 - \frac{b}{a}},\quad a = \max{(a, b)}, b = \min{(a, b)}`
    """

    def __init__(self, extracted_feature_name: str = 'eccentricity'):
        super().__init__(extracted_feature_name)

    def extract(self, X: pd.DataFrame) -> pd.Series:
        """Extract and return the eccentricity feature."""
        return np.sqrt(1. - X[['x', 'y']].min(axis=1)
                       / X[['x', 'y']].max(axis=1))


class TableDistanceExtractor(_FeatureExtractor):
    r"""Transformer, extract relative table distance from mean value.

    ``talbe`` feature is required. Formally the feature is defined
    as:

    :math:`t' =  |t - \mathbb{E}[t]|`
    """
    mean_value = 0.

    def __init__(self, extracted_feature_name: str = 'table_distance'):
        super().__init__(extracted_feature_name)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the extractor."""
        self.mean_value = X.table.mean()
        return self

    def extract(self, X: pd.DataFrame) -> pd.Series:
        """Extract and return the table distance feature."""
        return (X.table - self.mean_value).abs()


class DepthDistanceExtractor(_FeatureExtractor):
    r"""Transformer, extract relative table distance from mean value.

    ``talbe`` feature is required. Formally the feature is defined
    as:

    :math:`t' =  |t - \mathbb{E}[t]|`
    """
    mean_value = 0.

    def __init__(self, extracted_feature_name: str = 'depth_distance'):
        super().__init__(extracted_feature_name)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the extractor."""
        self.mean_value = X.depth.mean()
        return self

    def extract(self, X: pd.DataFrame) -> pd.Series:
        """Extract and return the depth distance feature."""
        return (X.depth - self.mean_value).abs()
