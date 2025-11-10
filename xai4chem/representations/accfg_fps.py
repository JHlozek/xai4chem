import os
import numpy as np
import pandas as pd
import joblib
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold
from xai4chem.tools.accfg.main import AccFG


REF_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(REF_PATH, "tools", "accfg")

def accfg_featurizer(smi_list):
    afg = AccFG()
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    fgs_ref = {fg : i for i, fg in enumerate(df["Functional Group"].tolist())}
    
    arr = np.zeros((len(smi_list), len(fgs_ref)))
    for i, s in enumerate(smi_list):
        fgs = afg.run(s, show_atoms=True)
        for f in fgs.keys():
            arr[i][fgs_ref[f]] += 1
    return arr


def accfg_explainer(smi_list):
    if type(smi_list) is not list:
        smi_list = [smi_list]
    afg = AccFG()
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    fgs_ref = {fg : i for i, fg in enumerate(df["Functional Group"].tolist())}

    arr = np.zeros((len(smi_list), len(fgs_ref)))
    bitInfo = []
    for i, s in enumerate(smi_list):
        fgs = afg.run(s, show_atoms=True)
        fgs_list = {}
        for f in fgs.keys():
            arr[i][fgs_ref[f]] += 1
            atoms = ()
            for val in fgs[f][0]:
                atoms = atoms + ((val,),)
            fgs_list[fgs_ref[f]] = atoms
        bitInfo.append(fgs_list)
    return arr, fgs_list

class AccFgFingerprint(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        X = accfg_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = accfg_featurizer(smiles)
        return pd.DataFrame(X, columns=["fp-{0}".format(i) for i in range(X.shape[1])])

    def explain_mols(self, smiles):
        X, bitInfo = accfg_explainer(smiles)
        return pd.DataFrame(X, columns=["fp-{0}".format(i) for i in range(X.shape[1])]), bitInfo
    
    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)
