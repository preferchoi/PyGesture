from fastapi import FastAPI
from Server.db import init_db

app = FastAPI()


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
def read_root():
    return {"Hello": "World"}
