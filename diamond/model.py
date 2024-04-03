"""Model definition, full pipeline definition and tools."""
from typing import TypeVar, Union, Any
from collections.abc import Sequence

import numpy as np
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from diamond import data

_DictOrListOfDicts = TypeVar('_DictOrListOfDicts',
                             bound=Union[dict[str, Any],
                                         list[dict[str, Any]]])


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


def make_params(user_param_grid: _DictOrListOfDicts,
                grid=True) -> _DictOrListOfDicts:
    """Make a sklearn-ready parameter dict from user parameters.

    A *user parameters* are a parameter format designed to be
    user readable. In practice, it is a simple superset of the sklearn
    expected format, where certain fields (dict keys) are preprocessed
    in order to transform the grid from user readable form to sklearn
    ready form.

    Currently applied transformations are::

    * The ``selector`` parameter. In the user params, the selector
        parameter shall be populated by a list of features
        (column names) sequences. Such a sequences is converted into a
        ``ColumnTransformer`` objects, which are compatible with the
        main pipeline (see :attr:`pipeline`).

    All other keys will not be transformed, so that all sklearn valid
    parameters which are specified by the user are passed down to the
    sklearn pipeline. Naturally, any valid sklearn parameters dict is
    also a valid user dict.

    If ``grid`` is ``True`` (default), then the input is expected to
    be a parameter *grid*, so that the output will be a grid transformed
    accordingly and ready to be used with sklearn grid search utilities.

    See the default user parameter grid
    :attr:`DEFAULT_USER_PARAMETER_GRID` for an example of user grid
    compatible with our main pipeline :attr:`pipeline`.

    TODO: Support lists of dicts as grids. Only pure grids are supported
    for the moment.
    """
    transformed_subdicts = {}
    if 'selector' in user_param_grid:
        if grid:
            trans_selector = _transform_selector_user_param(
                user_param_grid['selector'])
        else:
            trans_selector = data.make_feature_selector(
                *user_param_grid['selector'])

        transformed_subdicts['selector'] = trans_selector

    # Other eventual transformations... Changing the model? etc.

    return user_param_grid | transformed_subdicts


def _transform_selector_user_param(selectors: Sequence[Sequence[str]]
                                   ) -> tuple[ColumnTransformer, ...]:
    """Transform selector parameters from user grid to sklearn-ready.

    Used by :func:`make_param_grid` to do the transformation.
    """
    return tuple(map(lambda s: data.make_feature_selector(*s), selectors))
