from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
import pandas as pd
import numpy as np


import optuna.integration.lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


def create_features(df):
    mols = []
    for mol in df:
        mols.append(Chem.MolFromSmiles(mol))

    fingerprints = []
    for mol in mols:
        fingerprint1 = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        fingerprint2 = [x for x in AllChem.GetMACCSKeysFingerprint(mol)]
        fingerprints.append(np.append(fingerprint1, fingerprint2))
        
    fingerprints = np.array(fingerprints)

    return fingerprints
    

if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset.csv")
    X = create_features(df["SMILES"])
    y = df["IC50 (nM)"]
    y = np.log1p(y)
    
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X, y)

    pickle.dump(model, open(os.path.dirname(__file__) + "/model.pkl", "wb"))
