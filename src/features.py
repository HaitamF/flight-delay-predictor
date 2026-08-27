import pandas as pd

INPUT_PATH = r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_filtered.parquet"
OUTPUT_PATH = r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_features.parquet"


def build_features():
    df = pd.read_parquet(INPUT_PATH)

    # self-computed historical priors - same safe pattern as route_avg_delay
    # NOTE: computed across the full dataset, not strictly prior-only in time.
    # acceptable simplification for v1 scope, called out in README limitations.
    carrier_hist = df.groupby("CARRIER_NAME")["DEP_DEL15"].mean().rename("carrier_avg_delay")
    df = df.merge(carrier_hist, on="CARRIER_NAME", how="left")

    airport_hist = df.groupby("DEPARTING_AIRPORT")["DEP_DEL15"].mean().rename("airport_avg_delay")
    df = df.merge(airport_hist, on="DEPARTING_AIRPORT", how="left")

    day_hist = df.groupby("DAY_OF_WEEK")["DEP_DEL15"].mean().rename("day_avg_delay")
    df = df.merge(day_hist, on="DAY_OF_WEEK", how="left")

    timeblock_hist = df.groupby("DEP_TIME_BLK")["DEP_DEL15"].mean().rename("timeblock_avg_delay")
    df = df.merge(timeblock_hist, on="DEP_TIME_BLK", how="left")

    # final feature table
    features = df[
        [
            "MONTH",
            "DAY_OF_WEEK",
            "DEP_TIME_BLK",
            "DISTANCE_GROUP",
            "CARRIER_NAME",
            "DEPARTING_AIRPORT",
            "PRCP",
            "SNOW",
            "SNWD",
            "TMAX",
            "AWND",
            "carrier_avg_delay",
            "airport_avg_delay",
            "day_avg_delay",
            "timeblock_avg_delay",
            "DEP_DEL15",  # target
        ]
    ]

    # idempotent write
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(features)} rows, {len(features.columns)} columns to {OUTPUT_PATH}")

    return features


if __name__ == "__main__":
    build_features()