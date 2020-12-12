import sys
import os
import pickle
import pandas as pd

input_data = []
for line in sys.stdin:
    input_data.append(line.strip().split(","))

input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
input_df = input_df[["MaxEStateIndex", "MinEStateIndex"]]

input_df = input_df.fillna(0)

X = input_df
model = pickle.load(open(os.path.dirname(__file__) + "/model.pkl", "rb"))
y_pred = model.predict(X)

for val in y_pred:
    print(val)
