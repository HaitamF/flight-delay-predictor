"""
SHAP global feature importance for the flight delay model.

What this does:
1. Loads the trained model + saved category maps (from train.py)
2. Loads the processed data
3. Applies the SAME categorical -> code mapping used at training time
4. Runs SHAP's TreeExplainer (fast + exact for XGBoost)
5. Saves a global importance plot (bar chart) as PNG

Run from project root:
    python src/explain.py
"""

import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---- CONFIG: adjust this to your actual filename ----
PROCESSED_DATA_PATH = "C:\\Users\\hp\\OneDrive\\Bureau\\personal_projects\\flight-delay-predictor\\data\\processed\\flights_features.parquet"  # <-- change if different
MODEL_PATH = "C:\\Users\\hp\\OneDrive\\Bureau\\personal_projects\\flight-delay-predictor\\models/model.pkl"
ENCODERS_PATH = "C:\\Users\\hp\\OneDrive\\Bureau\\personal_projects\\flight-delay-predictor\\models/encoders.pkl"
TARGET_COL = "DEP_DEL15"
OUTPUT_PLOT = "C:\\Users\\hp\\OneDrive\\Bureau\\personal_projects\\flight-delay-predictor\\models/shap_global_importance.png"
SAMPLE_SIZE = 3000  # SHAP on a random sample is enough for global importance
RANDOM_STATE = 42
# -------------------------------------------------------

# 1. Load model + encoders
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    category_maps = pickle.load(f)

# 2. Load data (sampled — SHAP on millions of rows is extremely slow / memory heavy)
df = pd.read_parquet(PROCESSED_DATA_PATH)
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"Sampled {SAMPLE_SIZE} rows out of full dataset for SHAP (full run is too slow).")

# 3. Apply saved category mappings (same values -> same codes as training)
for map_name, mapping in category_maps.items():
    col = map_name.replace("_map", "").upper()
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Separate features from target
X = df.drop(columns=[TARGET_COL])

# 4. Run SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 5. Global importance plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Saved global feature importance plot to {OUTPUT_PLOT}")