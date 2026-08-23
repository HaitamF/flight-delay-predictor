import pandas as pd
KEEP_COLUMNS = [
    "FlightDate", "Airline", "Origin", "Dest", "CRSDepTime",
    "Distance", "Month", "DayOfWeek", "Cancelled", "Diverted", "DepDel15"
]

df = pd.read_parquet(
    r"C:\Users\hp\OneDrive\Bureau\personal_projects\flight-delay-predictor\data\raw\Combined_Flights_2019.parquet",
    columns=KEEP_COLUMNS
)

#filtering diverted and cancelled columns before dropping the columns
df = df[(~df["Cancelled"]) & (~df["Diverted"])]
df = df.drop(columns=["Cancelled", "Diverted"])
df = df.dropna(subset=["DepDel15"]) 
print(df.shape)
#filternig to 5 major airports based on their geographique and climates diversity and high  volume flights

airports=["ORD", "ATL", "JFK", "DFW", "LAX"]
df=df[df["Origin"].isin(airports) & df["Dest"].isin(airports)]

df.to_parquet("data/processed/flights_filtered.parquet", index=False)
print("Saved:", df.shape)