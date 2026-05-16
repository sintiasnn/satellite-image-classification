from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.routes import router
from src.models.classifier import classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier.load_model()
    yield


app = FastAPI(
    title="Satellite Image Classification API",
    description="API untuk klasifikasi citra satelit (cloudy, desert, green_area, water)",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Satellite Image Classification API",
        "docs": "/docs",
    }
