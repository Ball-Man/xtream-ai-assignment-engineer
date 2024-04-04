"""REST API web server controller."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Diamond model API"}
