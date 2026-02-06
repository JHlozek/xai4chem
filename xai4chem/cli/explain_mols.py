import os
import joblib
import pandas as pd
from xai4chem.supervised import Regressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def explain_mols(args):
    # Determine model type
    temp_model = Regressor(args.output_dir)
    temp_model.load_model(args.model_path)
    if hasattr(temp_model.model, 'predict_proba'):
        model = Classifier(args.output_dir)
        model_type = 'clf'
    else:
        model = Regressor(args.output_dir)
        model_type = 'reg'
    
    model.load_model(args.model_path)
    
    # Load and transform data
    data = pd.read_csv(args.input_file)
    smiles = data["smiles"]
    descriptor = model.descriptor
    features = descriptor.transform(smiles)
    
    # Make predictions and save results
    if model_type == 'reg':
        predictions = model.model_predict(features)
        pd.DataFrame({"smiles": smiles, "pred": predictions}).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
        
        # Plot score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions, kde=True)
        plt.title('Score Distribution Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(args.output_dir, 'score_distribution_plot.png'))
        plt.close()
    else:
        proba, pred = model.model_predict(features)
        pd.DataFrame({"smiles": smiles, "proba": proba, "pred": pred}).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
        
        # Plot score violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=pred, y=proba)
        plt.title('Score Violin Plot')
        plt.xlabel('Predicted Class')
        plt.ylabel('Predicted Probability')
        plt.savefig(os.path.join(args.output_dir, 'score_violin_plot.png'))
        plt.close()
        
        # Plot score strip plot
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=pred, y=proba, hue=pred)
        plt.title('Score Strip Plot')
        plt.xlabel('Predicted Class')
        plt.ylabel('Predicted Probability')
        plt.legend(title='Predicted Class', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(args.output_dir, 'score_strip_plot.png'))
        plt.close()
        
    if args.index_col is not None:
        ids = data[args.index_col]
        for i, smi in enumerate(tqdm(smiles)):
            model.explain_mol_atoms(smi, file_prefix=ids[i])
    else:
        for smi in tqdm(smiles):
            model.explain_mol_atoms(smi)
