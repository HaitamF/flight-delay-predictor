import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/flights_features.parquet"
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
TARGET_COL = "DEP_DEL15"

MODEL_FEATURES = [
    "MONTH", "DAY_OF_WEEK", "DEP_TIME_BLK", "DISTANCE_GROUP",
    "CARRIER_NAME", "DEPARTING_AIRPORT",
    "PRCP", "SNOW", "SNWD", "TMAX", "AWND",
    "carrier_avg_delay", "airport_avg_delay", "day_avg_delay", "timeblock_avg_delay",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    category_maps = pickle.load(f)

df = pd.read_parquet(DATA_PATH)
for map_name, mapping in category_maps.items():
    col = map_name.replace("_map", "").upper()
    if col in df.columns:
        df[col] = df[col].map(mapping)

X = df[MODEL_FEATURES]
y = df[TARGET_COL]

# Same split as train.py — so this is the real held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Save raw curve data
out = pd.DataFrame({
    "threshold": list(thresholds) + [1.0],
    "precision": precision,
    "recall": recall,
})
out.to_csv("pr_curve_data.csv", index=False)

# Print a few candidate thresholds
print("threshold | precision | recall")
for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
    idx = (thresholds >= t).argmax()
    print(f"{t:.2f}      | {precision[idx]:.4f}    | {recall[idx]:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (v2 model)")
plt.savefig("pr_curve.png", dpi=120)
print("\nSaved: pr_curve_data.csv, pr_curve.png")