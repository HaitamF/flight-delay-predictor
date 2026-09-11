import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # gradient boosted trees - tabular data, nonlinear interactions, no scaling/one-hot needed
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.xgboost
import json
import os
from sklearn.metrics import confusion_matrix

df = pd.read_parquet(r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_features.parquet")

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
# categorical columns in this feature set
categorical_cols = ["DEP_TIME_BLK", "CARRIER_NAME", "DEPARTING_AIRPORT"]

category_maps = {}
for col in categorical_cols:
    df[col] = df[col].astype("category")
    category_maps[col.lower() + "_map"] = dict(zip(df[col], df[col].cat.codes))
    df[col] = df[col].cat.codes

X = df.drop(columns=["DEP_DEL15","timeblock_avg_delay"])
y = df["DEP_DEL15"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale_pos_weight corrects for class imbalance (~21% delayed) - found necessary in v1 baseline
# ... (imports, data loading, categorical encoding, X/y split — unchanged) ...

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flight-delay-predictor")

with mlflow.start_run():
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("features", list(X.columns))
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    recall = report["1"]["recall"]
    precision = report["1"]["precision"]
    f1 = report["1"]["f1-score"]
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("recall_delayed", recall)
    mlflow.log_metric("precision_delayed", precision)
    mlflow.log_metric("f1_delayed", f1)
    mlflow.log_metric("roc_auc", auc)

    mlflow.xgboost.log_model(model, "model")
        # --- Write live model stats for the dashboard (real MLOps: no manual number-typing) ---
    cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
 
    # Preserve top_features from the last SHAP run if it exists, so this write doesn't erase it
    existing_top_features = []
    if os.path.exists("api/model_stats.json"):
        with open("api/model_stats.json", "r") as f:
            existing_top_features = json.load(f).get("top_features", [])
 
    model_stats = {
        "model_version": "v2",
        "features_used": len(X.columns),
        "recall_delayed": round(recall, 4),
        "precision_delayed": round(precision, 4),
        "accuracy": round(report["accuracy"], 4),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
        "top_features": existing_top_features,
    }
    with open("api/model_stats.json", "w") as f:
        json.dump(model_stats, f, indent=2)
 
    print("Updated api/model_stats.json with live training results.")

print(classification_report(y_test, y_pred))
print("ROC AUC:", auc)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(category_maps, f)

print("Model + encoders saved to models/")