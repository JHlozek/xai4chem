import os
import pandas as pd  
import sys
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt 


root = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.join(root, "..", "xai4chem"))

from datamol_desc import DatamolDescriptor
from regressor import Regressor

if __name__ == "__main__": 
    # Read data from CSV file into a DataFrame
    data = pd.read_csv(os.path.join(root, "..", "data", "plasmodium_falciparum_3d7_ic50.csv"))
    
    # Extract SMILES and target values
    smiles = data["smiles"]
    target = data["uM_value"] #uM_value or pchembl_value

    # Split data into training and test sets
    smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)
    
        
    # Instantiate the descriptor class 
    descriptor = DatamolDescriptor(discretize=False)

    descriptor.fit(smiles_train)
    
    # Transform the data 
    smiles_train = descriptor.transform(smiles_train)
     
    # Instantiate the regressor
    regressor = Regressor(algorithm='xgboost') 

    # Train the model 
    regressor.train(smiles_train, y_train)

     #Evaluate model
    # Transform    
    smiles_valid = descriptor.transform(smiles_valid)
    
    regressor.evaluate(smiles_valid, y_valid) 

    # Explain the model
    # Feature names
    feature_names = descriptor.feature_names
    
    explanation = regressor.explain(smiles_train) 
    
    new_explanation = shap.Explanation(
        values=explanation.values, 
        base_values=explanation.base_values, 
        data=explanation.data, 
        feature_names=feature_names
    )
    
    #waterfall plot
    shap_waterfall_plot = shap.plots.waterfall(new_explanation[0], max_display=15, show=False)
    shap_waterfall_plot.figure.savefig(os.path.join(root, "..", "results", "pf_3d7_ic50_waterfall_plot_explanation.png"),  bbox_inches='tight') 
    plt.close(shap_waterfall_plot.figure)
    
    #summary plot 
    shap_summary_plot = shap.plots.bar(new_explanation, max_display=20,  show=False)
    shap_summary_plot.figure.savefig(os.path.join(root, "..", "results", "pf_3d7_ic50_bar_plot_explanation.png"), bbox_inches='tight') 
    plt.close(shap_summary_plot.figure)
    
    #beeswarm plot
    shap_beeswarm_plot = shap.plots.beeswarm(new_explanation,max_display=15, show=False)
    shap_beeswarm_plot.figure.savefig(os.path.join(root, "..", "results", "pf_3d7_ic50_beeswarm_plot_explanation.png"), bbox_inches='tight') 
    plt.close(shap_beeswarm_plot.figure)