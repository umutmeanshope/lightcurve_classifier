import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data():

    # check if the files exist

    if all(os.path.exists(f"{f}.pkl") for f in ["X_train", "X_test", "y_train", "y_test"]):

        X_train = pd.read_pickle(f"X_train.pkl")
        X_test = pd.read_pickle(f"X_test.pkl")
        y_train = pd.read_pickle(f"y_train.pkl")
        y_test = pd.read_pickle(f"y_test.pkl")

    else:

        # Load the extracted features
        features = pd.read_pickle("extracted_features.pkl")

        # Separate features and target
        X = features.drop(columns=["tid", "target"])
        y = features["target"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save the splits for future use
        X_train.to_pickle(f"X_train.pkl")
        X_test.to_pickle(f"X_test.pkl")
        y_train.to_pickle(f"y_train.pkl")
        y_test.to_pickle(f"y_test.pkl")

    return X_train, X_test, y_train, y_test

split_data()