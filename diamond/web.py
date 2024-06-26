"""REST API web server controller."""
import os.path
import os
from glob import iglob
from typing import Optional, Annotated, Generic, TypeVar
import pickle
import uuid
from collections.abc import Iterable, Sequence

import aiocache
from aiocache.serializers import PickleSerializer
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from diamond import data
from diamond import model

_T = TypeVar('_T')

API_DESCRIPTION = '''\
This is a REST API to manage and serve diamond price prediction models.

Resources are structured in two main hierarchies: models and datasets.
Users can create and update datasets using the `/datasets/...` endpoints.
Similarly, new models can be trained and queried using the `/models/...`
endpoints.

To get started, get the available datasets with `GET /datasets` and
available models with `GET /models`. All GETs return paginated results.

> WARNING: these endpoints are a proof of concept and do not provide
extensive validation of the parameters, mostly due to time constraints.
'''
app = FastAPI(title='Diamond API', description=API_DESCRIPTION)

DATASETS_ROOT_LOCATION = os.path.join('datasets', 'diamonds')
MODELS_ROOT_LOCATION = 'models'

DATASET_CSV_HEADER = ('carat', 'cut', 'color', 'clarity', 'depth', 'table',
                      'price', 'x', 'y', 'z')
DATASET_DTYPES = {
    'carat': np.float32,
    'depth': np.float32,
    'table': np.float32,
    'x': np.float32,
    'y': np.float32,
    'z': np.float32,
}
DATASET_TARGET_DTYPE = {
    'price': np.float32
}


results_cache: aiocache.BaseCache = aiocache.Cache(
    serializer=PickleSerializer())


async def get_cache_paginated(cache: aiocache.BaseCache, key: str, page: int,
                              page_size: int) -> tuple[int, Sequence]:
    """Extract a value from cache and return it paginated.

    This approach always deserializes the entire cached value, which may
    not be ideal for scalability. Yet, it scales better than replicating
    IO operations or model predictions at each query.
    """
    cached_values = await cache.get(key)

    total_pages, remainder = divmod(len(cached_values), page_size)

    start_offset = page * page_size
    return (total_pages + (remainder != 0),
            cached_values[start_offset : start_offset + page_size])     # NOQA


async def cache_filesystem(cache: aiocache.BaseCache, key: str, directory: str,
                           format_: str):
    """Filter the given directory and cache results (not recursive).

    This completely replaces any previously cached values on the same
    key. A list of file basenames (without extension) is cached.
    """
    # Do this with a complex one liner if you seek performance
    resources = []
    for full_path in iglob(os.path.join(directory, f'*.{format_}')):
        resources.append(os.path.splitext(os.path.basename(full_path))[0])

    await cache.set(key, resources)


class QueryCache:
    """Cache for model queries."""

    def __init__(self, results_cache: aiocache.BaseCache):
        self._cache = aiocache.Cache()      # Only used for query IDs
        # Only used for data batches
        self._batches_cache = aiocache.Cache(serializer=PickleSerializer())
        # Must have a reference to the results cache in order to clean
        # up on delete.
        self._results_cache = results_cache

    async def get_batch_ids(self, id_: str, values=None) -> Iterable[str]:
        """Retrieve an iterable generating all batch ids for a query."""
        if values is None:
            values = await self._cache.get(id_)

        return map(lambda v: f'{id_}/{v}', range(values))

    async def exists(self, id_: str) -> bool:
        """Retrieve whether the given id exists."""
        return await self._cache.exists(id_)

    async def batches(self, id_: str) -> int:
        """Retrieve number of batches in the query."""
        return await self._cache.get(id_)

    async def delete(self, id_: str):
        """Delete the given query from cache, if it exists."""
        values = await self._cache.get(id_)

        # Iteratively delete all cache batches
        batch_ids = await self.get_batch_ids(id_, values)
        for batch_id in batch_ids:
            await self._batches_cache.delete(batch_id)

        # Finally delete master id
        await self._cache.delete(id_)

        # If results were computed, clean them as well
        is_results = await self._results_cache.exists(id_)
        if is_results:
            await self._results_cache.delete(id_)

    async def generate(self) -> str:
        """Generate a unique id and cache it.

        The generated id serves as a "master" location for the query.
        Multiple batches can be appended to the query, through
        :func:`add_batch`. See the function's docs for more info.
        """
        new_id = str(uuid.uuid4())
        exists = await self.exists(new_id)
        while exists:       # Conflicts are rare, keep trying
            new_id = str(uuid.uuid4())
            exists = await self.exists(new_id)

        await self._cache.set(new_id, 0)
        return new_id

    async def add_batch(self, id_: str, batch: np.ndarray):
        """Add a new batch to the cache.

        This creates a new key in the cache in the form:
        ``id_/x`` where ``x`` is ``cache[id_]``. Moreover, the internal
        counter ``cache[id_]`` is increased by one.
        """
        # Retrieve current count and increase it
        current_index = await self._cache.get(id_)
        await self._cache.increment(id_)

        await self._batches_cache.set(f'{id_}/{current_index}', batch)

    async def get_whole_batch(self, id_: str) -> np.ndarray:
        """Retrieve and concatenate all cached batches.

        Batches are not deleted from cache.
        """
        batch_ids = await self.get_batch_ids(id_)
        batches = await self._batches_cache.multi_get(list(batch_ids))
        return np.concatenate(batches)


query_cache = QueryCache(results_cache)


def get_model_location(id_, format_='pkl',
                       root=MODELS_ROOT_LOCATION) -> str:
    """Get model location on file system from resource ID."""
    return f'{os.path.join(root, id_)}.{format_}'


def get_dataset_location(id_, format_='csv',
                         root=DATASETS_ROOT_LOCATION) -> str:
    """Get dataset location on file system from resource ID."""
    # TODO: this replicates get_model_location, merge the functions
    #        and make wrappers.
    return f'{os.path.join(root, id_)}.{format_}'


def get_dataset(id_, format_='csv',
                root=DATASETS_ROOT_LOCATION,
                split_args={}) -> list[pd.DataFrame]:
    """Retrieve dataset splits from resource ID."""
    filepath = get_dataset_location(id_, format_, root)
    return data.split(*data.get_X_y(data.clean(data.load_raw(filepath))),
                      **split_args)


class Hyperparams(BaseModel):
    """ML model hyperparameters.

    Attribute names are designed to be sklearn-ready.
    """
    selector: list[str]
    linear__regressor__positive: bool

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'selector': ['carat', 'cut', 'color', 'clarity'],
                    'linear__regressor__positive': True
                }
            ]
        }
    }


class QueryLocation(BaseModel):
    """Prediction query descriptor."""
    location: str

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'location': '/models/example-model/example-query'
                }
            ]
        }
    }


class QuerySize(BaseModel):
    """Number of batches appended to the query so far."""
    batch_number: int

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'batch_numer': 3
                }
            ]
        }
    }


class DataSample(BaseModel):
    """A sample representing the properties of a single diamond."""
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float
    price: Optional[float] = None

    def numpy(self) -> np.ndarray:
        """Return data in a numpy array suitable for caching."""
        if self.price is None:
            return np.array([self.carat, self.cut, self.color, self.clarity,
                             self.depth, self.table, self.x, self.y, self.z])

        return np.array([self.carat, self.cut, self.color, self.clarity,
                         self.depth, self.table, self.price, self.x, self.y,
                         self.z])


class Prediction(BaseModel):
    """Diamond price prediction."""
    price: float

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'price': 2823
                }
            ]
        }
    }


class ResponsePage(BaseModel, Generic[_T]):
    """Paginated responses."""
    results: list[_T]
    next_page_location: Optional[str] = Field(
        examples=['/example_endpoint?page=2&page_size=1', None])
    total_pages: int = Field(examples=[10])


class MLModel(BaseModel):
    """Model metadata."""
    id_: str

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'id_': 'example_model'
                }
            ]
        }
    }


class Dataset(BaseModel):
    """Dataset metadata."""
    id_: str

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'id_': 'example_dataset'
                }
            ]
        }
    }


@app.exception_handler(Exception)
async def unicorn_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and create a response with its message.

    A simple solution, mostly for debug purposes.
    """
    return JSONResponse(
        status_code=500,
        content={'message': str(exc)},
    )


@app.on_event("startup")
async def setup_directories():
    """Create resource directories for models and datasets, if needed."""
    os.makedirs(DATASETS_ROOT_LOCATION, exist_ok=True)
    os.makedirs(MODELS_ROOT_LOCATION, exist_ok=True)

    await cache_filesystem(results_cache, 'models', MODELS_ROOT_LOCATION,
                           'pkl')
    await cache_filesystem(results_cache, 'datasets', DATASETS_ROOT_LOCATION,
                           'csv')


@app.get("/")
async def root():
    return {"message": "Diamond model API"}


@app.get("/datasets")
async def datasets_get(page: int, page_size: int) -> ResponsePage[Dataset]:
    """Get available dataset ids, paginated."""
    # TODO: this has replication with the models part, merge them
    is_results = await results_cache.exists('datasets')
    if not is_results:
        await cache_filesystem(results_cache, 'datasets',
                               DATASETS_ROOT_LOCATION, 'csv')

    total_pages, paginated_results = await get_cache_paginated(
        results_cache, 'datasets', page, page_size)

    next_ = None            # If this is the last page, return null value
    if page + 1 < total_pages:
        next_ = (f'/datasets?page={page + 1}&page_size={page_size}')
    return {'results': [{'id_': price} for price in paginated_results],
            'next_page_location': next_,
            'total_pages': total_pages}


@app.put("/datasets/{dataset_id}")
async def dataset_create(dataset_id: str):
    """Create or replace an existing dataset.

    The request will always create an empty dataset. Use
    ``POST /datasets/{dataset_id}`` to populate it, after being created.
    """
    await results_cache.delete('datasets')

    # Out of simplicity, use a preconfigured header for the csv
    with open(get_dataset_location(dataset_id), 'w') as fout:
        fout.write(','.join(DATASET_CSV_HEADER) + '\n')


@app.post("/datasets/{dataset_id}")
async def dataset_update(
    dataset_id: str,
    batch: Annotated[
        list[DataSample],
        Body(examples=[[{
                'carat': 1.1,
                'cut': 'Ideal',
                'color': 'H',
                'clarity': 'SI2',
                'depth': 62.0,
                'table': 55.0,
                'price': 4733.0,
                'x': 6.61,
                'y': 6.65,
                'z': 4.11
            },
            {
                'carat': 1.29,
                'cut': 'Ideal',
                'color': 'H',
                'clarity': 'SI1',
                'depth': 62.6,
                'table': 56.0,
                'price': 6424.0,
                'x': 6.96,
                'y': 6.93,
                'z': 4.35
            }
        ]])]):
    """Append a batch of data to a given dataset.

    The dataset must exist: create an empty one with
    ``PUT /datasets/{dataset_id}`` or retrieve an existing one with
    ``GET /datasets``. A batch of data is a list of samples, that is,
    of individual diamonds.
    """
    np_batch = np.vstack([sample.numpy() for sample in batch])

    df = (pd.DataFrame(np_batch,
                       columns=DATASET_CSV_HEADER)
          .astype(DATASET_DTYPES)
          .astype(DATASET_TARGET_DTYPE))

    # Datasets are csv, open in append mode and append without loading
    # it entirely.
    df.to_csv(get_dataset_location(dataset_id), mode='a', header=False,
              index=False)


@app.delete("/dataset/{dataset_id}")
async def dataset_delete(dataset_id: str):
    """Delete an existing dataset."""
    await results_cache.delete('datasets')

    dataset_location = get_dataset_location(dataset_id)

    os.remove(dataset_location)


@app.get("/models")
async def models_get(page: int, page_size: int) -> ResponsePage[MLModel]:
    """Get available model ids, paginated."""
    is_results = await results_cache.exists('models')
    if not is_results:
        await cache_filesystem(results_cache, 'models', MODELS_ROOT_LOCATION,
                               'pkl')

    total_pages, paginated_results = await get_cache_paginated(
        results_cache, 'models', page, page_size)

    next_ = None            # If this is the last page, return null value
    if page + 1 < total_pages:
        next_ = (f'/models?page={page + 1}&page_size={page_size}')
    return {'results': [{'id_': price} for price in paginated_results],
            'next_page_location': next_,
            'total_pages': total_pages}


@app.put("/models/{model_id}")
async def model_train(model_id: str, dataset_id: Annotated[str, Body()],
                      hyperparams: Optional[Hyperparams] = None):
    """Train a model using the given hyperparameters and dataset."""
    await results_cache.delete('models')

    X_train, X_test, y_train, y_test = get_dataset(dataset_id)

    model.pipeline.set_params(
        **model.make_params(hyperparams.dict(), grid=False))
    model.pipeline.fit(X_train, y_train)

    with open(get_model_location(model_id), 'wb') as fout:
        pickle.dump(model.pipeline, fout)


@app.delete("/models/{model_id}")
async def model_delete(model_id: str):
    """Delete an existing model."""
    await results_cache.delete('models')

    model_location = get_model_location(model_id)

    os.remove(model_location)


@app.post("/models/{model_id}/prices")
async def model_create_query(model_id: str) -> QueryLocation:
    """Create a prediction query for the model.

    The response contains an endpoint which can be used to populate
    the query. See the documentation of
    ``POST /models/{model_id}/prices/{query_id}``.
    """
    query_id = await query_cache.generate()

    return QueryLocation(location=f'models/{model_id}/prices/{query_id}')


@app.delete("/models/{model_id}/prices/{query_id}")
async def model_delete_query(model_id: str, query_id: str):
    """Create a prediction query for the model.

    The response contains an endpoint which can be used to populate
    the query. See the documentation of
    ``POST /models/{model_id}/prices/{query_id}``.
    """
    await query_cache.delete(query_id)


@app.post("/models/{model_id}/prices/{query_id}")
async def model_update_query(
    model_id: str,
    query_id: str,
    batch: Annotated[
        list[DataSample],
        Body(examples=[[{
                'carat': 1.1,
                'cut': 'Ideal',
                'color': 'H',
                'clarity': 'SI2',
                'depth': 62.0,
                'table': 55.0,
                'x': 6.61,
                'y': 6.65,
                'z': 4.11
            },
            {
                'carat': 1.29,
                'cut': 'Ideal',
                'color': 'H',
                'clarity': 'SI1',
                'depth': 62.6,
                'table': 56.0,
                'x': 6.96,
                'y': 6.93,
                'z': 4.35
            }
        ]])]) -> QuerySize:
    """Populate query for the model.

    Every request to this endpoint results in a new batch of data being
    appended to the specified query. Each batch is a list of samples,
    that is, a list of individual diamonds. When all the data has been
    appended, use ``GET /models{model_id}/prices/{query_id}`` to
    retrieve results.

    The response contains the total number of batches so far in the
    query.
    """
    numpy_batch = np.vstack([sample.numpy() for sample in batch])
    await query_cache.add_batch(query_id, numpy_batch)

    batch_number = await query_cache.batches(query_id)
    return {'batch_number': batch_number}


@app.get("/models/{model_id}/prices/{query_id}")
async def model_get_results(model_id: str, query_id: str,
                            page: int,
                            page_size: int) -> ResponsePage[Prediction]:
    """Get query results.

    Requesting this resource causes the model to predict the entirety
    of the query. The query must be first created through
    ``POST /models/{model_id}/prices`` and populated through
    ``POST /models/{model_id}/prices/{query_id}``. Query results are
    cached and can must accessed through pagination. After the result
    is computed, it is not possible to populate the query any further.

    Queries must be manually disposed via
    ``DELETE /models/{model_id}/prices/{query_id}``.
    """
    is_results = await results_cache.exists(query_id)

    # If not present, compute results and cache them
    if not is_results:
        with open(get_model_location(model_id), 'rb') as fin:
            loaded_model = pickle.load(fin)

        query_data = await query_cache.get_whole_batch(query_id)

        # TODO: this datatype conversions should happen directly in
        # the model pipeline.
        query_df = pd.DataFrame(query_data,
                                columns=loaded_model.feature_names_in_)
        query_df = query_df.astype(DATASET_DTYPES)
        results = loaded_model.predict(query_df)

        await results_cache.set(query_id, results)

    # There are a couple of extra cache reads here and there
    total_pages, paginated_results = await get_cache_paginated(
        results_cache, query_id, page, page_size)

    next_ = None            # If this is the last page, return null value
    if page + 1 < total_pages:
        next_ = (f'/models/{model_id}/prices/{query_id}'
                 f'?page={page + 1}&page_size={page_size}')
    return {'results': [{'price': price} for price in paginated_results],
            'next_page_location': next_,
            'total_pages': total_pages}
