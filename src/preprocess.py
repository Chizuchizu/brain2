# from features.base import Feature
# from features.base import Feature, generate_features, create_memo

# from src.pre_fun import base_data

# import cudf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from rdkit import Chem
from pathlib import Path
from mordred import Calculator, descriptors
from rdkit.Chem import rdMHFPFingerprint
from rdkit import RDLogger
from sklearn.decomposition import PCA
from numpy import inf

import os
import warnings

warnings.simplefilter("ignore")
RDLogger.DisableLog('rdApp.*')


# from mordred import Calculator, descriptors


# Feature.dir = "../features_data"
# data = pd.read_csv("../datasets/dataset.csv")

def pca_process(data, cwd):
    # print(data.info())
    # data = np.nan_to_num(data).astype(float)
    path = cwd / "../features/pca.pkl"
    if not os.path.isfile(path):
        data.columns = range(data.shape[1])
        for col in data.columns:
            # print(data[col].dtype)
            if data[col].dtype == "object" or data[col].dtype == "bool":
                data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

        # data[data == np.nan] = 0
        print(data.info())
        data = np.nan_to_num(data).astype(float)
        pca = PCA(n_components=500)
        data = pca.fit_transform(data)

        data = pd.DataFrame(data)

        data.to_pickle(path)
    else:
        data = pd.read_pickle(path)
    return data


def fe(data, cwd, train):

    cwd = Path(cwd)

    data["one_count_2"] = data["SMILES"].transform(lambda x: x.count("1")) == 2

    filepath = cwd / "../features/mordred_fe.pkl"
    if not os.path.isfile(filepath) or not train:
        # print("TRAIN_LOAD")
        data["SMILES"] = data["SMILES"].apply(
            lambda x: Chem.MolFromSmiles(x)
        )
        calc = Calculator(descriptors, ignore_3D=True)

        new_data = calc.pandas(data["SMILES"])

        if cwd != Path(""):
            new_data.to_pickle(filepath)
    else:
        new_data = pd.read_pickle(filepath)

    data = pd.concat(
        [
            data,
            new_data
        ],
        axis=1
    )

    filepath = cwd / "../features/finger_print.pkl"
    if not os.path.isfile(filepath) or not train:
        encoder = rdMHFPFingerprint.MHFPEncoder()

        # print("TRAIN_LOAD")
        if data["SMILES"].dtype == "object":
            data["SMILES"] = data["SMILES"].apply(
                lambda x: Chem.MolFromSmiles(x)
            )

        new_data = pd.concat(list(data["SMILES"].apply(lambda x: pd.DataFrame(encoder.EncodeMol(x)).T))).reset_index(
            drop=True)
        # new_data.columns = range(data.shape[1], data.shape[1] + new_data.shape[1])

        if cwd != Path(""):
            new_data.to_pickle(filepath)
    else:
        new_data = pd.read_pickle(filepath)
    # a = mordred_fe(data, cwd, train)
    data = pd.concat(
        [
            data,
            new_data
        ],
        axis=1
    )
    data = data.fillna(0)


    # カラム名は違えど要素が一緒のカラムは100個くらいあるけど気にしない（実行時間が長くなるので）
    # data = data.T.drop_duplicates().T
    # data.columns = range(data.shape[1])

    data = data.drop(
        columns=["SMILES"]
    )
    # target = data["target"]
    # data = pca_process(data[[col for col in data.columns if col != "target"]], cwd)
    # data["target"] = target.copy()

    return data


def run(cwd, data=False):
    if type(cwd) == str:
        cwd = Path(cwd)

    train = False
    if type(data) == bool:
        print("TRAIN")
        train = True
        data = pd.read_csv(cwd / "../datasets/dataset.csv")
        data = data.rename(
            columns={
                "log P (octanol-water)": "target"
            }
        )
    data = fe(data, cwd, train)

    if train:
        data.to_pickle(cwd / "../features/data_1.pkl")

    return data.astype(float)


if __name__ == "__main__":
    run(Path(""))
