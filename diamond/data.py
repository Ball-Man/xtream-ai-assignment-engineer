"""Data pipeline components."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

CUT_GRADES = 'Poor', 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'


def load_raw(file_or_path) -> pd.DataFrame:
    """Load raw dataset from file (csv only, so far)."""
    return pd.read_csv(file_or_path)


cut_grades_encoder = OrdinalEncoder(categories=[list(CUT_GRADES)],
                                    dtype=np.float32)
