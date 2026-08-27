"""
Request/response schemas for the flight delay prediction API.

FlightInput mirrors the 20 feature columns the model was trained on
(everything except DEP_DEL15, the target).
"""

from pydantic import BaseModel


class FlightInput(BaseModel):
    MONTH: int
    DAY_OF_WEEK: int
    DEP_TIME_BLK: str
    DISTANCE_GROUP: int
    CARRIER_NAME: str
    DEPARTING_AIRPORT: str
    PRCP: float
    SNOW: float
    SNWD: float
    TMAX: float
    AWND: float
    carrier_avg_delay: float
    airport_avg_delay: float
    day_avg_delay: float
    timeblock_avg_delay: float

    class Config:
        json_schema_extra = {
            "example": {
                "MONTH": 7,
                "DAY_OF_WEEK": 5,
                "DEP_TIME_BLK": "1700-1759",
                "DISTANCE_GROUP": 4,
                "CARRIER_NAME": "Delta Air Lines Inc.",
                "DEPARTING_AIRPORT": "Atlanta Municipal",
                "PRCP": 0.0,
                "SNOW": 0.0,
                "SNWD": 0.0,
                "TMAX": 88.0,
                "AWND": 7.5,
                "carrier_avg_delay": 0.21,
                "airport_avg_delay": 0.19,
                "day_avg_delay": 0.20,
                "timeblock_avg_delay": 0.25,
            }
        }


class PredictionOutput(BaseModel):
    delay_probability: float
    predicted_delayed: bool