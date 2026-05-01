import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.app.schemas.response_schema import PredictionResponse
from backend.app.services.inference_service import InferenceService
from backend.app.services.preprocessing_service import PreprocessingService

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["inference"])
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing uploaded file name.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max allowed size is {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    inference_service: InferenceService = request.app.state.inference_service
    preprocessing_service: PreprocessingService = request.app.state.preprocessing_service

    try:
        processed = preprocessing_service.preprocess_image_bytes(content)
        result = inference_service.predict(processed)
        return PredictionResponse(**result)
    except ValueError as exc:
        LOGGER.warning("Invalid image provided: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed.") from exc
