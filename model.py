import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from catboost import CatBoostRegressor

from utils import FEATURE_COLUMNS, CATEGORICAL_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rent_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "train_set.csv")
TARGET_COLUMN = "price"


class RentModel:
    def __init__(self, model):
        self.model = model

    def predict(self, features: pd.DataFrame):
        return self.model.predict(features)



def load_dataset():
    df = pd.read_csv(DATA_PATH)

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    df = df.drop_duplicates()

    df = df.dropna(subset=[TARGET_COLUMN])

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")

    numeric_columns = [
        "baths",
        "beds",
        "sqfeet",
        "lat",
        "long"
    ]

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    df = df[
        (df["price"] >= 300) &
        (df["price"] <= 8000)
    ]

    df = df[
        (df["sqfeet"] >= 250) &
        (df["sqfeet"] <= 5000)
    ]

    df = df[
        (df["beds"] >= 0) &
        (df["beds"] <= 8)
    ]

    df = df[
        (df["baths"] >= 1) &
        (df["baths"] <= 8)
    ]

    df = df[
        (df["lat"].between(18, 72)) &
        (df["long"].between(-180, -60))
    ]

    q_low = df["price"].quantile(0.02)
    q_high = df["price"].quantile(0.98)

    df = df[
        (df["price"] >= q_low) &
        (df["price"] <= q_high)
    ]

    return df



def train_model():
    print("Loading dataset...")

    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=10,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=200
    )

    print("Training CatBoost model...")

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLUMNS,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    return RentModel(model)



def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training new model...")
        return train_model()

    model = joblib.load(MODEL_PATH)

    return RentModel(model)



def get_feature_importance(model_wrapper):
    model = model_wrapper.model

    importance = model.get_feature_importance()

    return {
        feature: round(float(score), 4)
        for feature, score in zip(FEATURE_COLUMNS, importance)
    }



def predict_with_confidence(model_wrapper, features):
    prediction = float(model_wrapper.predict(features)[0])

    margin = prediction * 0.10

    return {
        "predicted_rent": round(prediction, 2),
        "min_rent": round(prediction - margin, 2),
        "max_rent": round(prediction + margin, 2)
    }


if __name__ == "__main__":
    train_model()