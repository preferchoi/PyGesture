from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from db import init_db

app = FastAPI()


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/gesture")
def get_gesture():
    
    return JSONResponse(content="", status_code=200)
