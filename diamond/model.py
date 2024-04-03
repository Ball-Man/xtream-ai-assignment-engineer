"""Model definition, full pipeline definition and tools."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from diamond import data


pipeline = Pipeline(
    steps=[
        ('encoder', data.sequential_encoder),
        ('volume_extractor', data.VolumeFeatureExtractor()),
        ('eccentricity_extractor', data.EccentricityFeatureExtractor()),
        ('table_extractor', data.TableDistanceExtractor()),
        ('depth_extractor', data.DepthDistanceExtractor()),
        ('log_transformer', data.column_log_transformer),
        ('scaler', StandardScaler()),
        ('selector', 'passthrough'),
        # Using data.log_transform as transformer here leads the same
        # results, however due to some internal functioning, it triggers
        # a warning. Hence, we prefer the latter form where log and exp
        # transformations are
        # ('linear', TransformedTargetRegressor(
        #     LinearRegression(), transformer=data.log_transformer))
        ('linear', TransformedTargetRegressor(
            LinearRegression(), func=np.log, inverse_func=np.exp))
    ]
)
pipeline.set_output(transform='pandas')
