"""Data pipeline components."""
from string import ascii_uppercase

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
