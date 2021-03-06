import sys
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import PolynomialFeatures
from rdkit.Chem import rdMolDescriptors

def smiles_to_num_of_element_and_bonds(smiles):
    res = []
    elements = ['P', 'Br', 'Ge', 'Fe', 'K', 'O', 'Pb', 'F', 'As', 'Si', 'Cl', 'B', 'He', 'Sn', 'Na', 'Pt', 'N+', 'N', 'Zn', 'H', 'Mn', 'S-', 'Al', 'Au', 'I', 'Hg', 'S', 'C', 'Se']
    for smile in smiles:
        tmp = []
        str_len = len(smile)
        bond_2 = smile.count("=")
        bond_3 = smile.count("#")
        mol = Chem.MolFromSmiles(smile)
        molformula = rdMolDescriptors.CalcMolFormula(mol)
        el_count = []
        for el in elements:
            el_count.append(molformula.count(el))
        tmp.extend([str_len, bond_2, bond_3])
        tmp.extend(el_count)
        res.append(tmp)
    return res

def create_fingerprints(df):
    mols = []
    for mol in df:
        mols.append(Chem.MolFromSmiles(mol))
    
    fingerprints = []
    for mol in mols:
        fingerprint2 = [x for x in AllChem.GetMACCSKeysFingerprint(mol)]
        fingerprints.append(fingerprint2)
        
    fingerprints = np.array(fingerprints)
    
    return fingerprints

if __name__ == "__main__":
    warnings.simplefilter("ignore")
        
    input_data = []
    for line in sys.stdin:
        input_data.append(line.strip().split(","))

    input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
    input_df = input_df.replace(r'^\s*$', np.nan, regex=True)
    
    train = input_df
    
    for col in train.columns:
        if col == "SMILES":
            continue
        train[col] = train[col].fillna(train[col].astype(float).mean())
    
    importance_gain_base44 = ['MolLogP', 'PEOE_VSA6', 'PEOE_VSA7', 'fr_benzene', 'MinPartialCharge', 'MaxAbsPartialCharge', 'NumHDonors', 'NHOHCount', 'TPSA', 'SlogP_VSA5', 'BalabanJ', 'fr_Al_COO', 'SMR_VSA10', 'SMR_VSA5', 'MinEStateIndex', 'NumAromaticCarbocycles', 'MinAbsEStateIndex', 'HallKierAlpha', 'EState_VSA5', 'SlogP_VSA1', 'MinAbsPartialCharge', 'MaxPartialCharge', 'MaxEStateIndex', 'PEOE_VSA8', 'EState_VSA8', 'VSA_EState10', 'PEOE_VSA1', 'FractionCSP3', 'MolMR', 'qed', 'PEOE_VSA2', 'BertzCT', 'Chi3v', 'SlogP_VSA3', 'VSA_EState9', 'NumHeteroatoms', 'Kappa3', 'FpDensityMorgan1', 'PEOE_VSA9', 'PEOE_VSA4', 'SlogP_VSA2', 'SlogP_VSA4', 'PEOE_VSA14', 'PEOE_VSA10']
    
    importance_gain_poly50 = [114, 16473, 3561, 4283, 3009, 8586, 4638, 16474, 16663, 2442, 2448, 13, 10625, 10163, 48, 2164, 13117, 3563, 10743, 414, 2346, 8541, 308, 5936, 9146, 14963, 8556, 7921, 11599, 8553, 11563, 2158, 1059, 296, 8738, 10451, 1098, 2633, 842, 1104, 7409, 2540, 11, 274, 2636, 1040, 11938, 6467, 7766, 202]
    
    X_base200 = train.drop(["SMILES"], axis=1)
        
    X_base44 = train[importance_gain_base44]
    
    X_fp = create_fingerprints(train["SMILES"])
    
    # 2????????????????????????
    poly = PolynomialFeatures(2)
    X_poly20000 = poly.fit_transform(X_base200)
    X_poly50 = X_poly20000[:,importance_gain_poly50] # gain??????50???
    
    # ?????????????????????????????????????????????
    X_count = smiles_to_num_of_element_and_bonds(train["SMILES"])
    
    # ??????????????????
    X_comb1 = np.append(X_base44, X_fp, axis=1)
    X_comb2 = np.append(X_comb1, X_poly50, axis=1)
    X_comb3 = np.append(X_comb2, X_count, axis=1)

    # ??????????????????
#    transformer = StandardScaler()
#    X_comb_scaled = transformer.fit_transform(X_comb3)
    scaler = pickle.load(open(os.path.dirname(__file__) + "/scaler.pkl", "rb"))
    X_comb_scaled = scaler.transform(X_comb3)
    

    model = pickle.load(open(os.path.dirname(__file__) + "/model.pkl", "rb"))
    y_pred = model.predict(X_comb_scaled)

    for val in y_pred:
        print(val)
