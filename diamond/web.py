"""REST API web server controller."""
import os.path
import os
from typing import Optional, Annotated
import pickle
import uuid
from collections.abc import Iterable

import aiocache
from aiocache.serializers import PickleSerializer
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diamond import data
from diamond import model

app = FastAPI()

DATASETS_ROOT_LOCATION = os.path.join('datasets', 'diamonds')
MODELS_ROOT_LOCATION = 'models'


class QueryCache:
    """Cache for unique query IDs."""

    def __init__(self):
        self._cache = aiocache.Cache()      # Only used for query IDs
        # Only used for data batches
        self._batches_cache = aiocache.Cache(serializer=PickleSerializer())

    async def get_batch_ids(self, id_: str, values=None) -> Iterable[str]:
        """Retrieve an iterable generating all batch ids for a query."""
        if values is None:
            values = await self._cache.get(id_)

        return map(lambda v: f'{id_}/{v}', range(values))

    async def exists(self, id_: str) -> bool:
        """Retrieve whether the given id exists."""
        return await self._cache.exists(id_)

    async def delete(self, id_: str):
        """Delete the given query from cache, if it exists."""
        values = await self._cache.get(id_)

        # Iteratively delete all cache batches
        batch_ids = await self.get_batch_ids(id_, values)
        for batch_id in batch_ids:
            await self._batches_cache.delete(batch_id)

        # Finally delete master id
        await self._cache.delete(id_)

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
        await self._cache.increase(id_)

        await self._batches_cache.set(f'{id_}/{current_index}', batch)

    async def get_whole_batch(self, id_: str) -> np.ndarray:
        """Retrieve and concatenate all cached batches.

        Batches are not deleted from cache.
        """
        batch_ids = await self.get_batch_ids(id_)
        batches = self._batches_cache.multi_get(list(batch_ids))
        return np.concatenate(batches)


query_cache = QueryCache()


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


@app.get("/")
async def root():
    return {"message": "Diamond model API"}


@app.put("/models/{model_id}")
async def model_train(model_id: str, dataset_id: Annotated[str, Body()],
                      hyperparams: Optional[Hyperparams] = None):
    """Train a model using the given hyperparameters and dataset."""
    X_train, X_test, y_train, y_test = get_dataset(dataset_id)

    model.pipeline.set_params(
        **model.make_params(hyperparams.dict(), grid=False))
    model.pipeline.fit(X_train, y_train)

    with open(get_model_location(model_id), 'wb') as fout:
        pickle.dump(model.pipeline, fout)


@app.delete("/models/{model_id}")
async def model_delete(model_id: str):
    """Delete an existing model."""
    model_location = get_model_location(model_id)

    os.remove(model_location)