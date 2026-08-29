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

Error Analysis — v2 Model (15 features)

Confusion breakdown on holdout set:

True Negatives: 1,732,145
False Positives: 995,891
True Positives: 484,334
False Negatives: 257,875

Finding: the model over-relies on time-of-day signal.

False positives (flagged delayed, actually on-time) concentrate in evening departure blocks (1500–1959), which have high timeblock_avg_delay (0.27–0.30) and above-average precipitation, snow, and wind.
False negatives (missed real delays) concentrate in morning departure blocks (0700–1159), which have low timeblock_avg_delay (0.11–0.19) and below-average bad weather.

Interpretation: the model has learned "evening + bad weather + high historical time-block rate → delayed" as a dominant rule. This is directionally correct but applied too bluntly — it over-triggers on evening flights that turn out fine, and under-triggers on morning flights that delay for reasons unrelated to time-of-day (e.g. mechanical, staffing, one-off disruptions). DEP_TIME_BLK and timeblock_avg_delay appear to be carrying disproportionate weight relative to other features, likely drowning out more flight-specific signal.

Carrier note: Southwest, American, and Delta appear frequently in both FP and FN breakdowns — consistent with them being the highest-volume carriers, not necessarily an airline-specific bias. Worth re-checking if this holds after normalizing for volume.