from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import sys
import os
import pickle
import pandas as pd
import numpy as np


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
    input_data = []
    for line in sys.stdin:
        input_data.append(line.strip())

    X = create_features(input_data)

    model = pickle.load(open(os.path.dirname(__file__) + "/model.pkl", "rb"))
    pred = model.predict(X)
    pred_exp = np.expm1(pred)

    for val in pred_exp:
        print(val)
