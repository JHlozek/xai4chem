# cli.py
import argparse
from .explain_global import explain_global
from .explain_mols import explain_mols

class XAI4ChemCLI:
    def __call__(self):
        parser = argparse.ArgumentParser(
            description="XAI4Chem CLI - A command-line interface for training and inference with XAI4Chem.",
        )

        subparsers = parser.add_subparsers(dest='command')

        # Global explanations command
        global_parser = subparsers.add_parser('explain_global', help='Train a model with the given input data and produce explanations for the model as a whole.')
        global_parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" and "activity" columns).')
        global_parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the trained model and evaluation reports.')
        global_parser.add_argument('-r', '--representation', type=str, required=True, choices=['datamol', 'morgan', 'accfg'],
                                  help='Type of molecular representation to use. Options are: datamol, morgan, accfg.')
        global_parser.set_defaults(func=explain_global)

        # Inference command
        mols_parser = subparsers.add_parser('explain_mols', help='Produce explanations for each individual molecule.')
        mols_parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the CSV file containing input data (must include "smiles" column).')
        mols_parser.add_argument('-m', '--model_dir', type=str, required=True, help='Directory containing the saved model file.')
        mols_parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the prediction results.')
        mols_parser.add_argument('-id_col', '--index_col', type=str, required=False, help='Column with smiles IDs for labeling output explanations.')
        mols_parser.set_defaults(func=explain_mols)

        args = parser.parse_args()
        args.func(args)

if __name__ == '__main__':
    cli = XAI4ChemCLI()
    cli()
