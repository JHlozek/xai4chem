import datamol as dm
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold


# To filter features with a high percentage of missing values.
class NanFilter:
    def __init__(self, max_na):
        self._name = "nan_filter"
        self.MAX_NA = max_na

    def fit(self, X):
        max_na = int((1 - self.MAX_NA) * X.shape[0])
        self.col_idxs = [j for j in range(X.shape[1]) if np.sum(np.isnan(X[:, j])) <= max_na]

    def transform(self, X):
        return X[:, self.col_idxs] 
    
    def save(self, file_name):
        joblib.dump(self, file_name)

    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)

# To impute missing values
class Imputer:
    def __init__(self):
        self._name = "imputer"
        self._fallback = 0

    def fit(self, X):
        self.impute_values = [np.median(X[:, j][~np.isnan(X[:, j])]) for j in range(X.shape[1])]

    def transform(self, X):
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = self.impute_values[j]
        return X
    
    def save(self, file_name):
        joblib.dump(self, file_name)

    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)

# To remove features that have almost constant values.
class VarianceFilter:
    def __init__(self):
        self._name = "variance_filter" 

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = self.sel.transform(np.arange(X.shape[1]).reshape(1, -1)).ravel()

    def transform(self, X):
        return self.sel.transform(X) 
    
    def save(self, file_name):
        joblib.dump(self, file_name)

    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)
    

class Scaler(object):
    def __init__(self):
        self._name = "scaler"
        self.abs_limit = 10 

    def fit(self, X): 
        self.scaler = RobustScaler() 
        self.scaler.fit_transform(X) 

    def transform(self, X): 
        X = self.scaler.transform(X) 
        return np.clip(X, -self.abs_limit, self.abs_limit)
    
    def save(self, file_name):
        joblib.dump(self, file_name)

    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)

def datamol_featurizer(smiles_list):
    R = []
    for smiles in tqdm(smiles_list):
        mol = dm.to_mol(smiles)
        descriptors = dm.descriptors.compute_many_descriptors(mol)
        R.append(descriptors)
    return pd.DataFrame(R)

class DatamolDescriptor:
    def __init__(self, max_na=0.1, use_scaling=False, discretize=True,n_bins=5, kbd_strategy='quantile'):
        """
        Parameters:
        - max_na: float, optional (default=0.1)
            Maximum allowed percentage of missing values in features. 
            Whether to apply feature scaling.
        - discretize: bool, optional (default=True)
            Whether to discretize features.
        - n_bins: int, optional (default=5)
            Number of bins used for discretization.
        - kbd_strategy: str, optional (default='quantile')
            Strategy used for binning. Options: 'uniform', 'quantile', 'kmeans'.
        """
        self.nan_filter = NanFilter(max_na=max_na)
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()  
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=kbd_strategy)
        self.discretize = discretize
        self.use_scaling = use_scaling

    def fit(self, smiles):
        df = datamol_featurizer(smiles) 
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        if self.discretize:
            self.discretizer.fit(X)
        if self.use_scaling:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        col_idxs = self.variance_filter.col_idxs
        feature_names = list(df.columns)
        self.feature_names = [feature_names[i] for i in col_idxs]

    def transform(self, smiles):
        df = datamol_featurizer(smiles)  
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        if self.discretize:
            X = self.discretizer.transform(X)
        if self.use_scaling:
            X = self.scaler.transform(X)
        return np.array(X, dtype=int)

    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)