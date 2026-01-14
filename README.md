# XAI4Chem

Explainable AI for Chemistry

## Installation
For the environment,
```bash
conda create -n xai4chem python=3.10 -y
conda activate xai4chem
```
Then install from GitHub:
```bash
python -m pip install git+https://github.com/ersilia-os/xai4chem.git 
```
 
## Usage
### CLI
#### Training Explainer
Use the following command to train the model and produce global explanations for the training data:

```bash
xai4chem explain_global --input_file <path_to_input_csv> --output_dir <output_directory> --representation <representation_type>
```
- <path_to_input_csv>: Path to the CSV file containing input data (must include "smiles" and "activity" columns).
- <output_directory>: Directory to save the trained model, evaluation and interpretability reports.
- <representation_type>: Type of molecular representation to use. Options are: datamol, morgan, accfg.


#### Inference
Use the following command to make predictions with a trained model for a shortlist of molecules:

```bash 
xai4chem explain_mols --input_file <path_to_input_csv> --model_dir <model_directory> --output_dir <output_directory>
```
- <path_to_input_csv>: Path to the CSV file containing input data (must include "smiles" column).
- <model_directory>: Directory containing the saved model file.
- <output_directory>: Directory to save the prediction results, and interpretability reports.
- <index_col>: (Optional) provide the column name to use for naming the output files for each molecule instead of using the SMILES string of each molecule.

If the explainer was trained with the accfg representation, then functional group explanations will be produced for each molecule.

### API

See notebook example in /notebooks/train_and_plot_interpretability.ipynb


#### Data
Read data file and split:
```python
import pandas as pd  
from sklearn.model_selection import train_test_split

data = pd.read_csv("plasmodium_falciparum_3d7_ic50.csv") #data path

# Extract SMILES and target values
smiles = data["smiles"]
target = data["pchembl_value"] #target value's column_name

# Split data into training and test sets
smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)

# Reset indices
smiles_train.reset_index(drop=True, inplace=True)
smiles_valid.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)
```
Calculate and transform descriptors:
Choose datamol, morgan or accfg 
```python
from xai4chem import DatamolDescriptor

descriptor = DatamolDescriptor(discretize=False)

# Fit the descriptor to training data
descriptor.fit(smiles_train)

# Transform the data
smiles_train_transformed = descriptor.transform(smiles_train)
smiles_valid_transformed = descriptor.transform(smiles_valid)
```

#### Model Training and Evaluation
The tool provides `Regressor` and `Classifier` classes for training and evaluating regression and classification models respectively using the XGBoost algorithm. You can train the model with default parameters or perform hyperparameter optimization using Optuna.

Also, you can specify the number of features(k) to use.
Feature selection will automatically select the relevant k features during training. 
```python
from xai4chem.supervised import Regressor, Classifier

regressor = Regressor(output_folder, k=100) #Specify the output folder where evaluation metrics and interpretability plots will be saved.

# Train the model
regressor.fit(smiles_train_transformed, y_train, default_params=False)

#you can save the trained model
#regressor.save_model('model_filename.joblib') #pass the filename

# Evaluate the model
regressor.evaluate(valid_features, smiles_valid, y_valid)
```
#### Model Interpretation
The `Regressor` class also provides functionality for interpreting model predictions. You can generate plots by;
```python
regressor.explain(train_features, smiles_list=smiles_train, fingerprints='rdkit') #fingerprints='rdkit' or 'morgan'
```

#### Substructure Importance for Individual Molecules

```regressor.explain_mol_atoms("CC(=O)Nc1ccc(O)cc1")
```

#### Other models.
To generate interpretability plots for a trained model, use;
```python
from xai4chem.reporting import explain_model

regressor = Regressor(output_folder)
regressor.load_model(model_path)
regressor.explain_mol_atoms("CC(=O)Nc1ccc(O)cc1")
 
```

