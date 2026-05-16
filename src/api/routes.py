from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from src.models.classifier import classifier
from src.utils.image_utils import preprocess_image
from src.schemas.prediction import PredictionResponse, ErrorResponse

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if classifier.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = preprocess_image(image)
        predicted_class, confidence = classifier.predict(image_array)
        return PredictionResponse(
            predicted_class=predicted_class, confidence_scores=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
