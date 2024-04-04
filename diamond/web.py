"""REST API web server controller."""
import os.path
import os

import pandas as pd
from fastapi import FastAPI

from diamond import data

app = FastAPI()

DATASETS_ROOT_LOCATION = os.path.join('datasets', 'diamonds')
MODELS_ROOT_LOCATION = 'models'


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


@app.on_event("startup")
async def setup_directories():
    """Create resource directories for models and datasets, if needed."""
    os.makedirs(DATASETS_ROOT_LOCATION, exist_ok=True)
    os.makedirs(MODELS_ROOT_LOCATION, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Diamond model API"}
