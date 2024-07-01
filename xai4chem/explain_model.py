import shap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps

def explain_model(model, X, smiles_list, output_folder):
    """
    Explain the predictions of a given model using SHAP and visualize SHAP values on chemical structures.

    Parameters:
    - model: A trained model (classifier or regressor).
    - X: The feature set used for explanation.
    - smiles_list: A list of SMILES strings corresponding to the rows in X.
    - output_folder: Folder to save the SHAP plots and chemical structure visualizations.
    
    Returns:
    - explanation: SHAP values explaining the model predictions.
    """
    
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Percentiles for waterfall plots
    percentiles = [0, 25, 50, 75, 100]
    sample_indices = np.percentile(range(X.shape[0]), percentiles).astype(int)

    for i, idx in enumerate(sample_indices):
        # Get the SMILES string for the current sample
        smiles = smiles_list[idx] if smiles_list is not None else f'Sample {idx}'
        
        # Waterfall plot with SMILES string as title
        shap.waterfall_plot(explanation[idx], max_display=15, show=False)
        plt.title(f"Molecule: {smiles}")
        # Save the plot
        plt.savefig(os.path.join(output_folder, f"interpretability_sample_p{percentiles[i]}.png"), bbox_inches='tight')
        plt.close()
    
    # Summary plot (bar plot)
    shap.plots.bar(explanation, max_display=20,  show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_bar_plot.png"), bbox_inches='tight')
    plt.close()
    
    # Beeswarm plot
    shap.plots.beeswarm(explanation,max_display=15, show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_beeswarm_plot.png"), bbox_inches='tight')
    plt.close()
    
    # Scatter plots for top 5 features
    shap_values = explanation.values
    feature_names = X.columns
    
    top_features = np.argsort(-np.abs(shap_values).mean(0))[:5]
    
    for feature in top_features:
        shap.plots.scatter(explanation[:, feature], show=False)
        plt.savefig(os.path.join(output_folder, f"interpretability_{feature_names[feature]}.png"), bbox_inches='tight')
        plt.close()
    
    # Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{name}' for name in feature_names])
    data_df = pd.DataFrame(X, columns=feature_names)
    combined_df = pd.concat([data_df, shap_df], axis=1)
    combined_df.to_csv(os.path.join(output_folder, 'shap_values.csv'), index=False)
    
    return explanation
