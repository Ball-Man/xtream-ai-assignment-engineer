"""Data pipeline components."""
import abc
from string import ascii_uppercase

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

CUT_GRADES = 'Poor', 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'
COLORS = tuple(reversed(ascii_uppercase[3:]))
CLARITIES = ('I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1',
             'IF', 'FL')


def load_raw(file_or_path) -> pd.DataFrame:
    """Load raw dataset from file (csv only, so far)."""
    return pd.read_csv(file_or_path)


cut_grades_encoder = OrdinalEncoder(categories=[list(CUT_GRADES)],
                                    dtype=np.float32)

color_encoder = OrdinalEncoder(categories=[list(COLORS)],
                               dtype=np.float32)

clarity_encoder = OrdinalEncoder(categories=[list(CLARITIES)],
                                 dtype=np.float32)


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
