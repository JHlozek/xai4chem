import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xai4chem.representations import DatamolDescriptor, RDKitDescriptor, MordredDescriptor, MorganFingerprint, RDKitFingerprint
from xai4chem.supervised import Regressor, Classifier 
import datetime

class XAI4ChemCLI:
    def __call__(self):
        parser = argparse.ArgumentParser(
            description="XAI4Chem CLI - A command-line interface for training and inference with XAI4Chem.",
        )

        subparsers = parser.add_subparsers(dest='command')

        # Training command
        train_parser = subparsers.add_parser('train', help='Train a model with the given input data.')
        train_parser.add_argument('--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" and "activity" columns).')
        train_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model and evaluation reports.')
        train_parser.add_argument('--representation', type=str, required=True, choices=['datamol_descriptor', 'rdkit_descriptor', 'mordred_descriptor', 'morgan_fingerprint', 'rdkit_fingerprint'],
                                  help='Type of molecular representation to use. Options are: datamol_descriptor, rdkit_descriptor, mordred_descriptor, morgan_fingerprint, rdkit_fingerprint.')
        train_parser.set_defaults(func=self.train)

        # Inference command
        infer_parser = subparsers.add_parser('infer', help='Make predictions with a trained model.')
        infer_parser.add_argument('--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" column).')
        infer_parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the saved model file.')
        infer_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the prediction results.')
        infer_parser.set_defaults(func=self.infer)

        args = parser.parse_args()
        args.func(args)  
        
    def _get_descriptor(self, representation):
        '''
        Returns the descriptor/fingerprint class, 
        the fingerprints used(None for descriptor features), 
        and maximum features to be selected
        '''
        if representation == 'datamol_descriptor':
            return DatamolDescriptor(), None, None
        elif representation == 'rdkit_descriptor':
            return RDKitDescriptor(), None, 64
        elif representation == 'mordred_descriptor':
            return MordredDescriptor(), None, 100
        elif representation == 'morgan_fingerprint':
            return MorganFingerprint(), 'morgan', 100
        elif representation == 'rdkit_fingerprint':
            return RDKitFingerprint(), 'rdkit', 100
        else:
            raise ValueError("Invalid representation type")

    def train(self, args):
        # Load data
        data = pd.read_csv(args.input_file)
        smiles = data["smiles"]
        target = data["activity"]
        
        # Check if the problem is binary classification
        is_binary_classification = target.nunique() == 2 and set(target.unique()) <= {0, 1}
        
        # Split data
        smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)
        
        # Reset indices
        smiles_train.reset_index(drop=True, inplace=True)
        smiles_valid.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_valid.reset_index(drop=True, inplace=True)

        # Choose feature representation
        descriptor, fingerprints, max_features = self._get_descriptor(args.representation)
        
        # Fit and transform
        descriptor.fit(smiles_train)
        train_features = descriptor.transform(smiles_train)
        valid_features = descriptor.transform(smiles_valid)
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(args.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Choose appropriate model
        if is_binary_classification: 
            print('...Classification.....\n', target.value_counts())
            model = Classifier(reports_dir, descriptor=descriptor, k=max_features)
        else: 
            print('...Regression.....')
            model = Regressor(reports_dir, descriptor=descriptor, k=max_features)
            
        # Train model
        model.fit(train_features, y_train)
        
        # Generate reports
        model.evaluate(valid_features, smiles_valid, y_valid)
        model.explain(train_features, smiles_list=smiles_train, fingerprints=fingerprints)
        
        # Retrain final model on all data
        print('.........Training Final Model.................')
        descriptor.fit(smiles)
        all_features = descriptor.transform(smiles)
        model.fit(all_features, target)

        # Save final model
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        model_filename = os.path.join(args.output_dir, f"model_{timestamp}.pkl")
        model.save_model(model_filename) 

    def infer(self, args):
        # Load model and data
        model_path = os.path.join(args.model_dir, [f for f in os.listdir(args.model_dir) if f.endswith(".pkl")][0])
        
        # Load the model to determine its type
        temp_model = Regressor(args.output_dir)  # Create a temporary instance to determine the model type
        temp_model.load_model(model_path)
        
        if temp_model.model_type == 'regressor':
            model_details = Regressor(args.output_dir)
        elif temp_model.model_type == 'classifier':
            model_details = Classifier(args.output_dir)
        
        model_details.load_model(model_path)
        
        data = pd.read_csv(args.input_file)
        smiles = data["smiles"]
        
        # Transform data using the same representation
        descriptor = model_details.descriptor
        features = descriptor.transform(smiles)
        
        # Make predictions
        if model_details.model_type == 'regressor':
            predictions = model_details.model.model_predict(features) 
            output_path = os.path.join(args.output_dir, "predictions.csv")
            pd.DataFrame({"smiles": smiles, "pred": predictions}).to_csv(output_path, index=False)
        
        elif model_details.model_type == 'classifier':
            proba, pred = model_details.model_predict(features) 
            output_path = os.path.join(args.output_dir, "predictions.csv")
            pd.DataFrame({
                "smiles": smiles,
                "proba": proba,
                "pred": pred
            }).to_csv(output_path, index=False)

if __name__ == '__main__':
    cli = XAI4ChemCLI()
    cli()
