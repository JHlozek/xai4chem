import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from featurewiz import FeatureWiz

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "xai4chem"))

from datamol_desc import DatamolDescriptor
from rdkitclassical_desc import RDkitClassicalDescriptor
from mordred_desc import MordredDescriptor
from regressor import Regressor

output_folder = os.path.join(root, "..", "assets")

if __name__ == "__main__":
    # Read data from CSV file into a DataFrame
    data = pd.read_csv(os.path.join(root, "..", "data", "plasmodium_falciparum_3d7_ic50.csv"))

    # Extract SMILES and target values
    smiles = data["smiles"]
    target = data["pchembl_value"]

    # Split data into training and test sets
    smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)

    # Reset indices
    smiles_train.reset_index(drop=True, inplace=True)
    smiles_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)

    # Instantiate the descriptor class
    descriptor = RDkitClassicalDescriptor()

    descriptor.fit(smiles_train)

    # Transform the data 
    smiles_train_transformed = descriptor.transform(smiles_train)
    smiles_valid_transformed = descriptor.transform(smiles_valid)

    # Feature Selection
    fwiz = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='',
                      dask_xgboost_flag=False, nrows=None, verbose=0)
    X_train_selected, _ = fwiz.fit_transform(smiles_train_transformed, y_train)
    X_test_selected = fwiz.transform(smiles_valid_transformed)
    # List of selected features
    print(fwiz.features)

    # Instantiate the regressor
    regressor = Regressor(output_folder, algorithm='xgboost')
    
    # Train the model 
    regressor.fit(X_train_selected, y_train, default_params=True)
    # regressor.save_model(os.path.join(root, "..", "results", 'xgboost_1.joblib'))

    # Evaluate model
    regressor.evaluate(X_test_selected, y_valid)

    # Explain the model     
    regressor.explain(X_train_selected)
