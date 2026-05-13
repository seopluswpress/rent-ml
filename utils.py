import pandas as pd

FEATURE_COLUMNS = [
    "baths",
    "beds",
    "sqfeet",
    "region",
    "state",
    "lat",
    "long",
    "type",
    "laundry_options",
    "parking_options",
    "cats_allowed",
    "dogs_allowed",
    "smoking_allowed",
    "wheelchair_access",
    "electric_vehicle_charge",
    "comes_furnished"
]

CATEGORICAL_COLUMNS = [
    "region",
    "state",
    "type",
    "laundry_options",
    "parking_options"
]


def preprocess_input(data: dict) -> pd.DataFrame:
    processed = {
        "baths": float(data["baths"]),
        "beds": int(data["beds"]),
        "sqfeet": int(data["sqfeet"]),
        "region": data["region"],
        "state": data["state"],
        "lat": float(data["lat"]),
        "long": float(data["long"]),
        "type": data["type"],
        "laundry_options": data["laundry_options"],
        "parking_options": data["parking_options"],
        "cats_allowed": int(data["cats_allowed"]),
        "dogs_allowed": int(data["dogs_allowed"]),
        "smoking_allowed": int(data["smoking_allowed"]),
        "wheelchair_access": int(data["wheelchair_access"]),
        "electric_vehicle_charge": int(data["electric_vehicle_charge"]),
        "comes_furnished": int(data["comes_furnished"]),
    }

    return pd.DataFrame([processed], columns=FEATURE_COLUMNS)