"""
Error analysis: breaks predictions into TP/FP/FN/TN and shows what
false positives and false negatives have in common.

Run from project root:
    python src/error_analysis.py
"""

import pickle
import pandas as pd

# ---- CONFIG ----
DATA_PATH = r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_features.parquet"
MODEL_PATH = r"models\model.pkl"
ENCODERS_PATH = r"models\encoders.pkl"
TARGET_COL = "DEP_DEL15"
THRESHOLD = 0.5
# -----------------

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    category_maps = pickle.load(f)

df = pd.read_parquet(DATA_PATH)
display_df = df.copy()

for map_name, mapping in category_maps.items():
    col = map_name.replace("_map", "").upper()
    if col in df.columns:
        df[col] = df[col].map(mapping)

X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

probs = model.predict_proba(X)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

display_df["y_true"] = y_true
display_df["y_pred"] = y_pred
display_df["delay_probability"] = probs

def label_case(row):
    if row.y_true == 1 and row.y_pred == 1:
        return "TP"
    if row.y_true == 0 and row.y_pred == 0:
        return "TN"
    if row.y_true == 0 and row.y_pred == 1:
        return "FP"
    return "FN"

display_df["case"] = display_df.apply(label_case, axis=1)

print("\nConfusion breakdown:")
print(display_df["case"].value_counts())

# What do false positives look like vs true negatives?
fp = display_df[display_df["case"] == "FP"]
fn = display_df[display_df["case"] == "FN"]

compare_cols = [c for c in [
    "CARRIER_NAME", "DEPARTING_AIRPORT", "DEP_TIME_BLK",
    "carrier_avg_delay", "airport_avg_delay", "day_avg_delay", "timeblock_avg_delay",
    "PRCP", "SNOW", "AWND"
] if c in display_df.columns]

print("\n--- False Positives: top values by frequency (flagged delayed, weren't) ---")
for col in compare_cols:
    if display_df[col].dtype == "object" or display_df[col].nunique() < 30:
        print(f"\n{col}:")
        print(fp[col].value_counts(normalize=True).head(5))
    else:
        print(f"\n{col} (avg in FP vs overall):")
        print(f"  FP mean: {fp[col].mean():.3f} | overall mean: {display_df[col].mean():.3f}")

print("\n--- False Negatives: top values by frequency (missed real delays) ---")
for col in compare_cols:
    if display_df[col].dtype == "object" or display_df[col].nunique() < 30:
        print(f"\n{col}:")
        print(fn[col].value_counts(normalize=True).head(5))
    else:
        print(f"\n{col} (avg in FN vs overall):")
        print(f"  FN mean: {fn[col].mean():.3f} | overall mean: {display_df[col].mean():.3f}")

# Save the full labeled set for later digging (e.g. Excel filter, plots)
display_df.to_parquet("data/processed/error_analysis.parquet")
print("\nSaved full labeled predictions to data/processed/error_analysis.parquet")