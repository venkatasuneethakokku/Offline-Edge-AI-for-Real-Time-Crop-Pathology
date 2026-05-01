import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes.predict import router as predict_router
from backend.app.services.inference_service import InferenceService
from backend.app.services.preprocessing_service import PreprocessingService


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = logging.getLogger("app.startup")

    try:
        # Base directory: backend/
        base_dir = Path(__file__).resolve().parents[1]

        model_path = Path(
            os.getenv("TFLITE_MODEL_PATH")
            or os.getenv("MODEL_PATH")
            or (base_dir / "ml" / "models" / "crop_model.tflite")
        )

        class_names_path = Path(
            os.getenv("CLASS_NAMES_PATH")
            or (base_dir / "ml" / "models" / "class_names.json")
        )

        disease_info_path = Path(
            os.getenv("DISEASE_INFO_PATH")
            or (base_dir / "app" / "data" / "disease_info.json")
        )

        logger.info("Model path: %s", model_path)
        logger.info("Class names path: %s", class_names_path)
        logger.info("Disease info path: %s", disease_info_path)

        inference_service = InferenceService(
            model_path,
            class_names_path,
            disease_info_path,
        )
        inference_service.load()

        app.state.inference_service = inference_service
        app.state.preprocessing_service = PreprocessingService(target_size=(224, 224))

        logger.info("Application startup complete")
        yield

    except Exception:
        logger.exception("Application failed during startup")
        raise

    finally:
        logger.info("Application shutdown complete")


app = FastAPI(
    title="Crop Disease Detection API",
    version="1.0.0",
    description="FastAPI inference backend for crop disease detection.",
    lifespan=lifespan,
)

# Configure CORS
origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(predict_router)

# Mount frontend static files
project_root = Path(__file__).resolve().parents[2]
frontend_path = project_root / "frontend"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="frontend")


@app.get("/")
async def serve_frontend():
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Crop Disease Detection API is running"}


@app.get("/health")
async def health_check():
    model_loaded = bool(
        getattr(app.state, "inference_service", None)
        and app.state.inference_service.is_loaded
    )
    return {
        "status": "ok",
        "model_loaded": model_loaded,
    }


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Request validation failed",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logging.getLogger("app.error").exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=19900,
        log_level="info",
    )