"""REST API web server controller."""
import os.path
import os
from typing import Optional, Annotated
import pickle
import uuid

import aiocache
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel

from diamond import data
from diamond import model

app = FastAPI()

DATASETS_ROOT_LOCATION = os.path.join('datasets', 'diamonds')
MODELS_ROOT_LOCATION = 'models'


class QueryCache:
    """Cache for unique query IDs."""

    def __init__(self):
        self._cache = aiocache.Cache()

    async def exists(self, id_: str) -> bool:
        """Retrieve whether the given id exists."""
        return await self._cache.exists(id_)

    async def delete(self, id_: str):
        """Delete the given query from cache, if it exists."""
        values = await self._cache.get(id_)

        # Iteratively delete all cache batches
        for value in range(values):
            await self._cache.delete(id_ + f'/{value}')

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

    async def add_batch(self, id_: str, batch):
        """TODO"""


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
