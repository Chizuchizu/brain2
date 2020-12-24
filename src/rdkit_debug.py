import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMHFPFingerprint
import swifter
from rdkit.Chem.AtomPairs.Sheridan import GetBTFingerprint

from rdkit import DataStructs
import time
import warnings

warnings.simplefilter("ignore")
# start = time.time()
#
# data = pd.read_csv("../datasets/dataset.csv").sample(1000)
# print(pd.read_csv("../datasets/dataset.csv").shape)
# morgans = []  # 算出した MF を格納
# no_error = []  # エラーなく計算できたIDを格納
# df = pd.DataFrame()
encoder = rdMHFPFingerprint.MHFPEncoder()
# # encoder2 = GetBTFingerprint()
#
# for idx, smile in enumerate(data['SMILES']):  # df1 に PCCDB データが pandas DataFrame 型で入ってる
#     mol = Chem.MolFromSmiles(smile)
#
#     mhfp = encoder.EncodeMol(mol)
#
#     # btf = GetBTFingerprint(mol).GetNonzeroElements()
#
#     df = pd.concat(
#         [
#             df,
#             # pd.DataFrame.from_dict(fp, orient="index").T,
#             pd.DataFrame(mhfp).T
#
#         ]
#     )
#
# df = df.fillna(0)
#
# print(time.time() - start)

start = time.time()
data = pd.read_csv("../datasets/dataset.csv")#.sample(1000)
print(data["SMILES"].dtype)
print(data["SMILES"].dtype == "object")
data["SMILES"] = data["SMILES"].swifter.apply(lambda x: Chem.MolFromSmiles(x))
memo = pd.concat(list(data["SMILES"].swifter.apply(lambda x: pd.DataFrame(encoder.EncodeMol(x)).T)))
# print(pd.concat(memo))
print(time.time() - start)
#     no_error.append(idx)
#     morgans.append(fp.GetNonzeroElements())
#
# morgan_keys = []  # 全化合物のMFキーを格納
# for morgan in morgans:
#     morgan_keys += morgan.keys()
# morgan_keys = list(set(morgan_keys))
#
# morgan_mat = np.zeros((len(morgans), len(morgan_keys)))  # 全化合物のMF行列
# for idx1, morgan in enumerate(morgans):
#     for k, v in morgan.items():
#         morgan_mat[idx1][morgan_keys.index(k)] = v
print()
