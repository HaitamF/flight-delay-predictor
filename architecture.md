
flight-delay-predictor/
├── data/
│   ├── raw/              # OpenSky + Eurocontrol + weather pulls, untouched
│   └── processed/        # cleaned, joined, feature-engineered table
├── src/
│   ├── data_ingestion.py # pulls/loads raw sources
│   ├── features.py       # joins + engineered columns (route_avg_delay, etc.)
│   ├── train.py          # trains + saves model
│   └── predict.py        # loads model, scores new input
├── api/
│   ├── main.py            # FastAPI app (/predict, /health)
│   └── schemas.py         # Pydantic request/response models
├── models/
│   └── model.pkl          # trained model + model_card.md
├── notebooks/
│   └── eda.ipynb          # exploration only, not pipeline code
├── tests/
│   └── test_api.py
├── monitoring/
│   └── logs.py            # logs predictions for drift tracking
├── Dockerfile
├── docker-compose.yml      # if using a db
├── requirements.txt
├── .github/workflows/ci.yml
├── .gitignore
├── architecture.md         # already have this
└── README.md