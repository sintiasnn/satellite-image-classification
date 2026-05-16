from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_scores: dict[str, float]


class ErrorResponse(BaseModel):
    error: str
