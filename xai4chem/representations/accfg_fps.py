import os
import numpy as np
import pandas as pd
import joblib
from importlib import resources
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold
from xai4chem.tools.accfg.main import AccFG


REF_PATH = resources.files("xai4chem").joinpath("tools/accfg")

def ref_features():
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    fgs_ref = {fg : i for i, fg in enumerate(df["Functional Group"].tolist())}
    return fgs_ref
    
def ref_smarts():
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    smarts_ref = {sm : i for i, sm in enumerate(df["SMARTS Pattern"].tolist())}
    return smarts_ref

def accfg_featurizer(smi_list):
    afg = AccFG()
    fgs_ref = ref_features()
    
    arr = np.zeros((len(smi_list), len(fgs_ref)))
    for i, s in enumerate(smi_list):
        fgs = afg.run(s, show_atoms=True)
        for f in fgs.keys():
            for group in fgs[f]:
                arr[i][fgs_ref[f]] += 1
    return arr


def accfg_bit_explainer(smi_list):
    if type(smi_list) is not list:
        smi_list = [smi_list]
    afg = AccFG()
    fgs_ref = ref_features()

    arr = np.zeros((len(smi_list), len(fgs_ref)))
    bitInfo = []
    for i, s in enumerate(smi_list):
        fgs = afg.run(s, show_atoms=True)
        fgs_list = {}
        for f in fgs.keys():
            atoms = ()
            for group in fgs[f]:
                arr[i][fgs_ref[f]] += 1
                for val in group:
                    atoms = atoms + ((val,),)
            fgs_list[fgs_ref[f]] = atoms
        bitInfo.append(fgs_list)
    return arr, bitInfo

class AccFgFingerprint(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        X = accfg_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = accfg_featurizer(smiles)
        return pd.DataFrame(X, columns=self.features)

    def explain_mols(self, smiles):
        X, bitInfo = accfg_bit_explainer(smiles)
        return pd.DataFrame(X, columns=self.features), bitInfo[0]
    
    def get_ref_features(self):
        return ref_features()
    
    def get_ref_smarts(self):
        return ref_smarts()
    
    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)
