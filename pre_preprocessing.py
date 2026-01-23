import pandas as pd
from utils import batch_processing 

# THIS WILL OVERWRITE PREVIOUS FEATURE EXTRACTION FILEs

toi_data = pd.read_csv("toi_data.csv")

df = toi_data[["tid", "tfopwg_disp"]]

clean_df = df[df["tfopwg_disp"].isin(["CP", "FP"])].copy()

clean_df["target"] = clean_df["tfopwg_disp"].map({"CP": 1, "FP": 0})

# This will take sometime to run

features: pd.DataFrame = batch_processing(clean_df)

print("Batch processing complete.")

# save to pickle
print("Saving to pickle...")
features.to_pickle(f"extracted_features.pkl")