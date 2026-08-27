import pandas as pd

RAW_PATH = r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\raw\train.csv"
OUTPUT_PATH = r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_filtered.parquet"

# columns kept - see cahier de charge / architecture notes for reasoning per column
# dropped: SEGMENT_NUMBER, LATITUDE, LONGITUDE, AVG_MONTHLY_PASS_* (redundant with FLIGHTS_MONTH),
# FLT_ATTENDANTS_PER_PASS, GROUND_SERV_PER_PASS (low value, airline-constant)
# dropped entirely: CARRIER_HISTORICAL, DEP_AIRPORT_HIST, DAY_HISTORICAL, DEP_BLOCK_HIST
#   -> these are recomputed ourselves in features.py to guarantee no leakage
KEEP_COLUMNS = [
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
    "DEP_DEL15",  # target
]


def load_and_clean():
    df = pd.read_csv(RAW_PATH, usecols=KEEP_COLUMNS)

    # drop rows with missing target - can't train/evaluate without a label
    df = df.dropna(subset=["DEP_DEL15"])

    # basic cleaning: drop rows with missing values in core weather/operational fields
    # (small % expected, safe to drop rather than impute for v1)
    core_cols = ["PRCP", "SNOW", "SNWD", "TMAX", "AWND"]
    df = df.dropna(subset=core_cols)
    
    # drop exact duplicate rows if any
    df = df.drop_duplicates()

    # idempotent write - always overwrite, never append
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    load_and_clean()