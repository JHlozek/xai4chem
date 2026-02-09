import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xai4chem.representations import DatamolDescriptor, MorganFingerprint, AccFgFingerprint
from xai4chem.supervised import Regressor

def _get_descriptor(representation):
    '''
    Returns the descriptor/fingerprint class, 
    the fingerprints used (None for descriptor features), 
    and maximum features to be selected
    '''
    if representation == 'datamol':
        return DatamolDescriptor(), 22
    elif representation == 'accfg':
        return AccFgFingerprint(), 500
    elif representation == 'morgan':
        return MorganFingerprint(), 1024
    else:
        raise ValueError("Invalid representation type")

def explain_global(args):
    # Load data
    data = pd.read_csv(args.input_file)
    smiles = data["smiles"]
    target = data["activity"]
    
    # Check if the problem is binary classification
    is_binary_classification = target.nunique() == 2 and set(target.unique()) <= {0, 1}

    # Choose feature representation
    descriptor, max_features = _get_descriptor(args.representation)
    
    # Create directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Choose appropriate model
    if is_binary_classification: 
        print('...Classification.....\n', target.value_counts())
        active_percentage = (target[target == 1].count() / len(target)) * 100
        inactive_percentage = (target[target == 0].count()/ len(target)) * 100
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x=target)
        ax.set_xticklabels(['Inactive', 'Active'])
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        
        plt.title(f'Value Counts of Actives ({active_percentage:.0f}%) and Inactives ({inactive_percentage:.0f}%)')
        plt.ylabel('No. of Compounds')        
        plt.savefig(os.path.join(args.output_dir, 'dataset_distribution.png'))
        plt.close()
        model = Classifier(args.output_dir, fingerprints=args.representation, algorithm='xgboost', k=max_features)
    else: 
        print('...Regression.....')
        plt.figure(figsize=(8, 6))
        sns.histplot(target, kde=True, color='blue')
        plt.title('Distribution of Target Values')
        plt.xlabel('Target Values')
        plt.ylabel('No. of Compounds')
        plt.savefig(os.path.join(args.output_dir, 'dataset_distribution.png'))
        plt.close()
        model = Regressor(args.output_dir, fingerprints=args.representation, k=max_features)
        
    # Fit and transform
    all_features = descriptor.fit(smiles)
    model.fit(all_features, target)

    # Generate reports
    model.evaluate(all_features, smiles, target)
    model.explain(smiles)    

    # Save final model 
    model_filename = os.path.join(args.output_dir, "model_" + args.representation + ".pkl")
    model.save_model(model_filename)
   
