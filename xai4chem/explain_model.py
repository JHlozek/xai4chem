import shap
import os
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, X, output_folder):
    """
    Explain the predictions of a given model using SHAP.

    Parameters:
    - model: A trained model (classifier or regressor).
    - X: The feature set used for explanation.
    - output_folder: Folder to save the SHAP plots.
    """
    
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

     #waterfall plot
    waterfall_plot = shap.plots.waterfall(explanation[0], max_display=15, show=False)
    waterfall_plot.figure.savefig(os.path.join(output_folder, "interpretability_sample1.png"), bbox_inches='tight') 
    plt.close(waterfall_plot.figure)
    
    #summary plot 
    summary_plot = shap.plots.bar(explanation, max_display=20,  show=False)
    summary_plot.figure.savefig(os.path.join(output_folder, "interpretability_bar_plot.png"), bbox_inches='tight') 
    plt.close(summary_plot.figure)
    
    #beeswarm plot
    beeswarm_plot = shap.plots.beeswarm(explanation,max_display=15, show=False)
    beeswarm_plot.figure.savefig(os.path.join(output_folder, "interpretability_beeswarm_plot.png"), bbox_inches='tight') 
    plt.close(beeswarm_plot.figure)
    
    return explanation