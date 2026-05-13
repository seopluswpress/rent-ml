from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import (
    load_model,
    get_feature_importance,
    predict_with_confidence
)

from utils import preprocess_input

app = FastAPI(title="Rental Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()


class PredictionRequest(BaseModel):
    baths: float
    beds: int
    sqfeet: int
    region: str
    state: str
    lat: float
    long: float
    type: str
    laundry_options: str
    parking_options: str
    cats_allowed: int
    dogs_allowed: int
    smoking_allowed: int
    wheelchair_access: int
    electric_vehicle_charge: int
    comes_furnished: int


@app.get("/")
def home():
    return {
        "message": "Rental Price Prediction API Running"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    features = preprocess_input(request.model_dump())

    result = predict_with_confidence(model, features)

    return result


@app.get("/feature-importance")
def feature_importance():
    return get_feature_importance(model)