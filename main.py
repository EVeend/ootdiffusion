from fastapi import FastAPI
from ootd_router import router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.include_router(router, prefix="/ootd")