"""
Quick batch predictions using the trained model.

What this does:
1. Loads model + saved category maps
2. Loads processed data (or swap in a new CSV/parquet of flights)
3. Applies same encoding used at training time
4. Predicts delay probability for each flight
5. Prints the top N most likely delayed flights

Later, once the live API is wired up, this same load+encode+predict logic
moves into api/main.py for real-time scoring instead of batch.

Run from project root:
    python src/predict.py
"""

import pickle
import pandas as pd

# ---- CONFIG ----
DATA_PATH = "data/processed/flights_features.parquet"
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
TARGET_COL = "DEP_DEL15"  # will be dropped if present (e.g. scoring training data)
TOP_N = 15
# -----------------

# 1. Load model + encoders
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    category_maps = pickle.load(f)

# 2. Load data
df = pd.read_parquet(DATA_PATH)

# Keep a readable copy for display before we encode
display_df = df.copy()

# 3. Apply saved category mappings (same as training)
for map_name, mapping in category_maps.items():
    col = map_name.replace("_map", "").upper()
    if col in df.columns:
        df[col] = df[col].map(mapping)
        
# --- NEW: evening + bad weather interaction feature ---
EVENING_BLOCKS = [
    "1500-1559", "1600-1659", "1700-1759", "1800-1859", "1900-1959",
]
df["evening_bad_weather"] = (
    df["DEP_TIME_BLK"].isin(EVENING_BLOCKS) &
    ((df["PRCP"] > 0.1) | (df["SNOW"] > 0.1))
).astype(int)
# --- end new block ---

categorical_cols = ["DEP_TIME_BLK", "CARRIER_NAME", "DEPARTING_AIRPORT"]
# The 15 features the v2 model was trained on — order matters for XGBoost
MODEL_FEATURES = [
    "MONTH", "DAY_OF_WEEK", "DEP_TIME_BLK", "DISTANCE_GROUP",
    "CARRIER_NAME", "DEPARTING_AIRPORT",
    "PRCP", "SNOW", "SNWD", "TMAX", "AWND",
    "carrier_avg_delay", "airport_avg_delay", "day_avg_delay", "timeblock_avg_delay",
]

# 4. Build feature matrix — select only the columns the model was trained on
X = df[MODEL_FEATURES]

# 5. Predict probabilities for "delayed" class
probs = model.predict_proba(X)[:, 1]
display_df["delay_probability"] = probs
display_df["predicted_delayed"] = (probs >= 0.5).astype(int)

# 6. Show top N most likely delays
top = display_df.sort_values("delay_probability", ascending=False).head(TOP_N)
cols_to_show = [c for c in ["CARRIER_NAME", "DEPARTING_AIRPORT", "PREVIOUS_AIRPORT",
                             "DEP_TIME_BLK", "MONTH", "DAY_OF_WEEK",
                             "delay_probability", "predicted_delayed"] if c in top.columns]

pd.set_option("display.width", 120)
print(f"\nTop {TOP_N} most likely delayed flights:\n")
print(top[cols_to_show].to_string(index=False))