import pandas as pd

df= pd.read_parquet("data/processed/flights_filtered.parquet")
#feauture engineering
df["dep_hour"] = (df["CRSDepTime"] // 100).astype(int)
#time normalization 
route_avg = df.groupby(["Origin", "Dest"])["DepDel15"].mean().rename("route_avg_delay")
df = df.merge(route_avg, on=["Origin", "Dest"], how="left")

features = df[["Origin", "Dest", "Airline", "dep_hour", "Month",
                "DayOfWeek", "Distance", "route_avg_delay", "DepDel15"]]

df.to_parquet("data/processed/flights_features.parquet", index=False)
print("Saved:", df.shape)