# from features.base import Feature
# from features.base import Feature, generate_features, create_memo

# from src.pre_fun import base_data

# import cudf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from rdkit import Chem


# from mordred import Calculator, descriptors


# Feature.dir = "../features_data"
# data = pd.read_csv("../datasets/dataset.csv")

def fe(data):
    data = data.drop(
        columns=["SMILES"]
    )
    return data


def run(cwd, data=False):
    train = False
    if type(data) == bool:
        train = True
        data = pd.read_csv(cwd / "../datasets/dataset.csv")
        data = data.rename(
            columns={
                "log P (octanol-water)": "target"
            }
        )
    data = fe(data)

    if train:
        data.to_csv(cwd / "../features/data_1.pkl")

    return data.astype(float)


if __name__ == "__main__":
    data = pd.read_csv("../datasets/dataset.csv")
    run(data)
