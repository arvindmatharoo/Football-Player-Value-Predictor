from pydantic import BaseModel

class PlayerInput(BaseModel):
    age_years: float
    min : float
    availability_ratio : float
    offensive_efficiency : float
    progressive_actions: float
    position : str
    compition : str

class PredictionResponse(BaseModel):
    predicted_value_tier: str
