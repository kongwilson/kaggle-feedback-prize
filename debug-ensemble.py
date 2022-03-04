"""
DESCRIPTION

Copyright (C) Weicong Kong, 3/03/2022
"""
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"


df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))

# WKNOTE: pd.get_dummies - pandas's one-hot encoder
dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
dfx = dfx[cols].copy()

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels = [c for c in dfx.columns if c != "id"]
dfx_labels = dfx[labels]
dfx["kfold"] = -1

for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
    print(len(trn_), len(val_))
    dfx.loc[val_, "kfold"] = fold

df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
print(df.kfold.value_counts())
print(df.pivot_table(index='kfold', values='id', aggfunc=lambda x: len(np.unique(x))))
print(df.pivot_table(index='kfold', columns='discourse_type', values='id', aggfunc=len))
df.to_csv("train_folds.csv", index=False)

