"""Test model module (unit tests).

For simplicity, test cases are not documented through docstrings. Yet,
they should be semantically very clear.
"""
from collections.abc import Sequence

import pytest
from sklearn.compose import ColumnTransformer

from context import diamond         # NOQA
from diamond import model


@pytest.fixture(scope='module', autouse=True)
def pipeline_pandas_output():
    model.pipeline.set_output(transform='pandas')


@pytest.mark.parametrize('user_selectors', (
    (),
    (('a',), ('a', 'b'), ('a', 'b', 'c')),
    [('a',), ('a', 'b'), ('a', 'b', 'c')],
    [['a',], ['a', 'b'], ['a', 'b', 'c']]
))
def test_transform_selector_user_param(user_selectors):
    trans_selectors = model._transform_selector_user_param(user_selectors)

    assert isinstance(trans_selectors, tuple)
    assert len(trans_selectors) == len(user_selectors)

    for user_sel, trans_sel in zip(user_selectors, trans_selectors):
        assert isinstance(trans_sel, ColumnTransformer)


@pytest.mark.parametrize('user_grid, grid', (
    ({}, True),
    ({'a': (1, 2, 3), 'b': ()}, True),
    ({'selector': ('a', 'b', 'c'), 'b': ()}, False),
    ({'selector': (('a',), ('a', 'b')), 'b': ()}, True)
))
def test_make_params(user_grid, grid):
    trans_params = model.make_params(user_grid, grid)

    for param, user_value in user_grid.items():
        if param == 'selector':         # Check later
            continue

        assert user_value == trans_params[param]

    # Test special cases: selector
    trans_selector = trans_params.get('selector')
    if trans_selector is None:
        return

    if grid:
        assert isinstance(trans_selector, Sequence)
        for selector in trans_selector:
            assert isinstance(selector, ColumnTransformer)
        return

    # If not grid
    assert isinstance(trans_selector, ColumnTransformer)
