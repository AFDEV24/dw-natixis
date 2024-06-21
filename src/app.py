from fastapi import FastAPI

from src import ENV
from src.routers import create_embeddings, query
from src.utils.middleware.auth import AuthMiddleware

app = FastAPI()
app.add_middleware(AuthMiddleware, api_key=ENV["API_KEY"])
app.include_router(create_embeddings.router)
app.include_router(query.router)
