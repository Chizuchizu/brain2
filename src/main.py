import sys
import joblib
import pandas as pd
import numpy as np
# from preprocess import run
import yaml

debug = True
if debug:
    from preprocess import run
else:
    from preprocess import run

input_data = []
for line in sys.stdin:
    input_data.append(line.strip().split(","))

input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
data = input_df
data = run("", data)#.astype(float)

filename = "config/training.yaml" if debug else "config.yaml"
with open(filename, "r+") as f:
    cfg = yaml.load(f)

pred = np.zeros(data.shape[0])
for fold in range(1, cfg["base"]["n_folds"] + 1):
    path = f"../models/{fold}.pkl" if debug else f"{fold}.pkl"
    estimator = joblib.load(path)

    pred += estimator.predict(data) / cfg["base"]["n_folds"]

for val in pred:
    print(val)
