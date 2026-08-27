# Flight Delay Predictor

## Project Goal

Predicts whether a flight will depart at least 15 minutes late. Target: `DEP_DEL15`
- `0`: on-time (delay under 15 min)
- `1`: delayed (delay ≥ 15 min)

Developed incrementally. This README covers work through the V1 baseline model.

## Dataset

**Source:** Kaggle — `threnjen/2019-airline-delays-and-cancellations`
**Time period:** 2019
**Rows used (train+test):** 3,470,245

**Columns:**
`MONTH, DAY_OF_WEEK, DEP_TIME_BLK, DISTANCE_GROUP, CARRIER_NAME,  DEPARTING_AIRPORT,  PRCP, SNOW, SNWD, TMAX, AWND, carrier_avg_delay, airport_avg_delay, day_avg_delay, timeblock_avg_delay, DEP_DEL15`

`carrier_avg_delay`, `airport_avg_delay`, `day_avg_delay`, `timeblock_avg_delay` are self-computed historical priors — recomputed independently rather than using the dataset's built-in equivalents, to guarantee no leakage into the target.

**Class distribution:**
- Class 0 (on-time): ~81.0%
- Class 1 (delayed): ~19.0% — imbalance handled via `scale_pos_weight`

## Performance — Test / Hold-out Set
v1 :
| Metric | Class 0 (On-time) | Class 1 (Delayed) | Overall |
|---|---|---|---|
| Precision | 0.89 | 0.31 | – |
| Recall | 0.67 | 0.64 | – |
| F1-score | 0.76 | 0.42 | – |
| Support | 732,691 | 171,667 | 904,358 |
| Accuracy | – | – | 0.66 |
| Macro avg | 0.60 | 0.65 | 0.59 |
| Weighted avg | 0.78 | 0.66 | 0.70 |

**Reading the results:**
- Strong precision on the majority (on-time) class.
- Recall on delayed flights (0.64) is meaningfully better than precision (0.31), a direct result of `scale_pos_weight` correcting for class imbalance — the model prioritizes catching delays over avoiding false alarms.
- Overall accuracy (0.66) is moderate but not the right metric to optimize here, given the imbalance — macro F1 (0.59) better reflects that the model still struggles specifically on the delayed class.
- Weather features (PRCP, SNOW, SNWD) are zero-inflated — most flights occur under clear conditions, limiting their standalone predictive value. `CONCURRENT_FLIGHTS` and the historical-average features appear to carry more signal.

v2: 

| Metric | Class 0 (On-time) | Class 1 (Delayed) | Overall |
|---|---|---|---|
| Precision | 0.87 | 0.33 | – |
| Recall | 0.63 | 0.65 | – |
| F1-score | 0.73 | 0.43 | – |
| Support | 545,697 | 148,442 | 694,049 |
| Accuracy | – | – | 0.64 |
| Macro avg | 0.60 | 0.64 | 0.58 |
| Weighted avg | 0.75 | 0.64 | 0.67 |

Model Performance

The classifier was evaluated on delay class recall/precision rather than raw accuracy, since the dataset is imbalanced (~81% on-time, ~19% delayed) and accuracy alone would reward a model that just predicts "on-time" every time.

v1 (21 features, including engineered priors + operational features like plane age, concurrent flights, and monthly flight volumes):

Delayed class: recall 0.64, precision 0.31

v2 (15 features — dropped PLANE_AGE, CONCURRENT_FLIGHTS, AIRPORT_FLIGHTS_MONTH, AIRLINE_FLIGHTS_MONTH, AIRLINE_AIRPORT_FLIGHTS_MONTH, PREVIOUS_AIRPORT):

Delayed class: recall 0.65, precision 0.33
Overall: accuracy 0.64, macro F1 0.58, weighted F1 0.67

Why features were dropped: the 6 removed columns are not realistically available at prediction time for a live user-facing API — they require either aircraft-specific data (tail number history, plane age) or real-time flight-traffic infrastructure (concurrent flights, live monthly volumes) that isn't accessible without a paid aviation data provider. Rather than imputing default values for unavailable inputs — which would silently degrade prediction integrity — the feature set was reduced to only what can be reliably sourced from user input, precomputed historical priors, and a free weather API (Open-Meteo).

Result: performance is effectively unchanged between v1 and v2 (recall 0.64→0.65, precision 0.31→0.33), indicating the dropped features carried little marginal signal beyond what the four historical-average priors (carrier, airport, day, timeblock) already capture. This validates prioritizing deployability over marginal feature completeness.

Interpretation of current numbers: recall of 0.65 means the model catches roughly two-thirds of actual delays — useful for a "heads up, this flight might be late" tool. Precision of 0.33 means about 1 in 3 flights flagged as delayed will actually be delayed — a meaningful false-alarm rate, documented here as a known limitation rather than treated as a bug, and a candidate for further tuning (threshold adjustment, class weighting) in future iterations.


**Next steps:** represent weather as a severity flag rather than a continuous value, evaluate performance specifically on the weather-affected subset, and revisit `scale_pos_weight` / decision threshold jointly rather than independently.


API (enrichment + basic prediction logging) → MLflow → SHAP + error analysis → Docker → tests → CI/CD → deployment