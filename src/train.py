import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # gradient boosted trees - tabular data, nonlinear interactions, no scaling/one-hot needed
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.xgboost

df = pd.read_parquet(r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\processed\flights_features.parquet")

# categorical columns in this feature set
categorical_cols = ["DEP_TIME_BLK", "CARRIER_NAME", "DEPARTING_AIRPORT"]

category_maps = {}
for col in categorical_cols:
    df[col] = df[col].astype("category")
    category_maps[col.lower() + "_map"] = dict(zip(df[col], df[col].cat.codes))
    df[col] = df[col].cat.codes

X = df.drop(columns=["DEP_DEL15"])
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

print(classification_report(y_test, y_pred))
print("ROC AUC:", auc)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(category_maps, f)

print("Model + encoders saved to models/")