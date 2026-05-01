from typing import List

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    disease_name: str = Field(..., description="Predicted disease class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    description: str = Field(..., description="Disease description")
    symptoms: List[str] = Field(default_factory=list, description="Common symptoms")
    treatment: List[str] = Field(default_factory=list, description="Recommended treatment steps")
    prevention: List[str] = Field(default_factory=list, description="Prevention guidance")
