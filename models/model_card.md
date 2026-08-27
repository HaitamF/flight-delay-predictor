Model Card — Flight Delay Predictor (v1)
Model

XGBoost binary classifier (XGBClassifier)

Hyperparameters
n_estimators: 200
max_depth: 6
learning_rate: 0.1
eval_metric: logloss
scale_pos_weight: ratio of class 0 / class 1 in training set
random_state: 42
Training setup
Split: 80/20 train/test, stratified on target
Categorical encoding: .cat.codes (integer codes, no one-hot — trees split on raw categories natively)
Encoders/category maps saved separately (models/encoders.pkl) for consistent encoding at inference
Inputs excluded (leakage)

Any column only known after departure: actual times, arrival delay, taxi times, delay cause codes. Cancelled/diverted flights dropped from the dataset entirely.

Version

v1 — baseline, no hyperparameter tuning performed yet.

Files
models/model.pkl — trained model
models/encoders.pkl — category code mappings + historical averages lookup