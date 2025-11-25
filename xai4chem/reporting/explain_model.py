import shap
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from rdkit.Geometry import Point2D
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
import xgboost as xgb
from xai4chem.representations.accfg_fps import AccFgFingerprint

RADIUS = 3
NBITS = 2048
_MIN_PATH_LEN = 1
_MAX_PATH_LEN = 7 


def explain_model(model, X, smiles_list, output_folder, fingerprints=None):
    print('Explaining model')
    create_output_folder(output_folder)
        
    X_cols = X.columns.tolist()
    X_matrix = xgb.DMatrix(X)
    
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    explanation.feature_names = X_cols
    
    #Samples
    predictions = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
    percentiles = [0, 25, 50, 75, 100]
    percentile_values = np.percentile(predictions, percentiles)

    # Indices of the predictions closest to the percentile values
    sample_indices = [np.argmin(np.abs(predictions - value)) for value in percentile_values]

    if fingerprints == "morgan" or fingerprints == "accfg":
        #scale SHAP values per atom
        raw_scores = []
        for idx, smiles in enumerate(smiles_list):
            bit_info, valid_top_bits, bit_shap_values = explain_mol_features(explainer, X_matrix.slice([idx]), smiles, X_cols, fingerprints)
            tmp = shapley_raw_total_per_atom(bit_info, bit_shap_values, smiles, fingerprints)
            raw_scores += list(tmp.values())
        scaler = MaxAbsScaler()
        scaler.fit(np.array(raw_scores).reshape(-1,1))

        
        for i, idx in enumerate(sample_indices):
            smiles = smiles_list[idx] if smiles_list is not None else f'Sample {idx}'
            plot_waterfall(explanation, idx, smiles, output_folder, f"interpretability_sample_p{percentiles[i]}.png", fingerprints)
    
            bit_info, valid_top_bits, bit_shap_values = explain_mol_features(explainer, X_matrix.slice([idx]), smiles, X_cols, fingerprints)
            atom_shapley_values = shapley_raw_total_per_atom(bit_info, bit_shap_values, smiles, fingerprints)
            scaled_shapley_values = scaler.transform(np.array(list(atom_shapley_values.values())).reshape(-1,1)).flatten()
            atom_shapley_values = {k:scaled_shapley_values[i] for i,k in enumerate(atom_shapley_values)}
            
            if fingerprints == "morgan":
                draw_top_features(bit_info, valid_top_bits, smiles,
                            os.path.join(output_folder, f'sample_p{percentiles[i]}_top_features.png'), fingerprints)
            # add feature drawing for AccFG fingerprints
            highlight_and_draw_molecule(atom_shapley_values, smiles,
                        os.path.join(output_folder, f"sample_p{percentiles[i]}_shap_highlights.png"))

    else:
        scaler = None

    if fingerprints == "accfg":
        ref_features = list(AccFgFingerprint().get_ref_features().keys())
        explanation_tmp = copy.deepcopy(explanation)
        accfg_feat_names = [ref_features[int(feat.split('-')[1])] for feat in explanation.feature_names]
        explanation_tmp.feature_names = accfg_feat_names
    else:
        explanation_tmp = explanation
    
    #save_shap_values_to_csv(explanation_tmp, X, X_cols, output_folder)
    plot_summary_plots(explanation_tmp, output_folder)
    plot_scatter_plots(explanation_tmp, output_folder)

    return explanation, explainer, scaler

def explain_mol_features(explainer, X, smiles, feature_names, fingerprints=None):
    if smiles is not None and fingerprints is not None:          
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))) # canonical smiles
        if mol:
            sample_shap_values = explainer.shap_values(X)[0]

            bit_info = {}
            bit_shap_values = {}
            if fingerprints == 'morgan':
                AllChem.GetHashedMorganFingerprint(mol, radius=RADIUS, nBits=NBITS, bitInfo=bit_info)   
            elif fingerprints == 'accfg':
                fp, bit_info = AccFgFingerprint().explain_mols(smiles)
            elif fingerprints == 'rdkit':
                Chem.RDKFingerprint(mol, minPath=_MIN_PATH_LEN, maxPath=_MAX_PATH_LEN, fpSize=NBITS, bitInfo=bit_info)
            
            for bit_idx, feature_name in enumerate(feature_names):
                bit = int(feature_name.split('-')[1])
                if 0 <= bit_idx < len(sample_shap_values):
                    bit_shap_values[bit] = sample_shap_values[bit_idx]

            valid_top_bits = [bit for bit in sorted(bit_shap_values.keys(), key=lambda b: abs(bit_shap_values[b]),
                                                    reverse=True) if bit in bit_info][:5]

            return bit_info, valid_top_bits, bit_shap_values            

def shapley_raw_total_per_atom(bit_info, bit_shap_values, smiles, fingerprints):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))) # canonical smiles
    atom_shapley_scores = {}
    if fingerprints == "accfg":
        for bit in bit_info:
            if bit in bit_shap_values:
                bit_shap_score = bit_shap_values[bit]
            else:
                continue
        
            for atom in bit_info[bit]:
                if atom[0] not in atom_shapley_scores:
                    atom_shapley_scores[atom[0]] = bit_shap_score
                else:
                    atom_shapley_scores[atom[0]] += bit_shap_score
        
    if fingerprints == "morgan":
        for bit in bit_info:
            if bit in bit_shap_values:
                bit_shap_score = bit_shap_values[bit]
            else:
                continue

            for centres in bit_info[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, centres[1], centres[0])
                atoms = set([centres[0]])
                atoms = list(atoms)
                for atom in atoms:
                    if atom not in atom_shapley_scores:
                        atom_shapley_scores[atom] = bit_shap_score
                    else:
                        atom_shapley_scores[atom] += bit_shap_score
                        
    #scale values with the MinMaxScaler before mapping to colours
    return atom_shapley_scores

def create_output_folder(output_folder):
    """Create output folder if it does not exist."""
    os.makedirs(output_folder, exist_ok=True)


def save_shap_values_to_csv(explanation, X, feature_names, output_folder, fingerprints='morgan'):
    """Save SHAP values and features to a CSV file."""
    shap_df = pd.DataFrame(explanation.values, columns=[f'SHAP_{name}' for name in feature_names])
    data_df = pd.DataFrame(X, columns=feature_names)
    combined_df = pd.concat([data_df, shap_df], axis=1)
    combined_df.to_csv(os.path.join(output_folder, 'shap_values-' + fingerprints + '.csv'), index=False)


def plot_waterfall(explanation, idx, smiles, output_folder, file_name, fingerprints):
    """Create a waterfall plot for a given sample."""   
    
    if fingerprints == "accfg":
        ref_features = list(AccFgFingerprint().get_ref_features().keys())
        explanation_tmp = copy.deepcopy(explanation)
        accfg_feat_names = [ref_features[int(feat.split('-')[1])] for feat in explanation.feature_names]
        explanation_tmp.feature_names = accfg_feat_names
        shap.plots.waterfall(explanation_tmp[idx], max_display=15, show=False)
    else:
        shap.plots.waterfall(explanation[idx], max_display=15, show=False)
    plt.title(f"Molecule: {smiles}")
    plt.savefig(os.path.join(output_folder, file_name), bbox_inches='tight')
    plt.close()


def plot_summary_plots(explanation, output_folder):
    """Create summary plots: bar plot and beeswarm plot."""       
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_bar_plot.png"), bbox_inches='tight')
    plt.close()

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_beeswarm_plot.png"), bbox_inches='tight')
    plt.close()


def plot_scatter_plots(explanation, output_folder):
    """Create scatter plots for the top 5 features."""
    shap_values = explanation.values
    top_features = np.argsort(-np.abs(shap_values).mean(0))[:5]

    for i, feature in enumerate(top_features):
        shap.plots.scatter(explanation[:, feature], show=False)
        plt.savefig(os.path.join(output_folder, f"interpretability_feature{i+1}.png"), bbox_inches='tight')
        plt.close()


def draw_top_features(bit_info, valid_top_bits, smiles, output_path, fingerprints):
    """Draw and save top features(bits)."""
    list_bits = []
    legends = []

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))) # canonical smiles
    for x in valid_top_bits:
            for i in range(len(bit_info[x])):
                list_bits.append((mol, x, bit_info, i))
                legends.append(str(x))
    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.prepareMolsBeforeDrawing = False    
    if fingerprints == 'morgan':
        p = Draw.DrawMorganBits(list_bits, molsPerRow=6, legends=legends, drawOptions=options)
        p.save(output_path)
    elif fingerprints == 'rdkit':
        p = Draw.DrawRDKitBits(list_bits, molsPerRow=6, legends=legends, drawOptions=options)
        p.save(output_path)

    add_title_to_image(output_path, f"Top 5 features({fingerprints}-fps)")


def highlight_and_draw_molecule(atoms_shapley_dict, smiles, output_path):
    for atom in atoms_shapley_dict:
        if atoms_shapley_dict[atom] > 1.0:
            atoms_shapley_dict[atom] = 1
        elif atoms_shapley_dict[atom] < -1.0:
            atoms_shapley_dict[atom] = -1

    min_val = min(atoms_shapley_dict.values())
    max_val = max(atoms_shapley_dict.values())
    max_abs = max(abs(min_val), abs(max_val))

    #Scale and color atoms
    atom_colors = {}
    colormap = custom_cmap()
    norm = plt.Normalize(-1*max_abs, max_abs)
    for atom in atoms_shapley_dict:
        c = colormap(norm(atoms_shapley_dict[atom]))
        c = (c[0], c[1], c[2], 0.7)
        atom_colors[atom] = c

    # Get the atom positions for drawing (this requires 2D coordinates)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))) # canonical smiles
    AllChem.Compute2DCoords(mol)

    #Color bonds by adjacent atoms
    bond_colors = {}
    #color_steps = 3
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetIdx() not in atom_colors or atom2.GetIdx() not in atom_colors:
            bond_colors[bond.GetIdx()] = (1,1,1,0.7)
        elif atoms_shapley_dict[atom1.GetIdx()] >= atoms_shapley_dict[atom2.GetIdx()]:
            bond_colors[bond.GetIdx()] = atom_colors[atom1.GetIdx()]
        else:
            bond_colors[bond.GetIdx()] = atom_colors[atom2.GetIdx()]

    drawer = MolDraw2DCairo(500, 500)
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors,
                        highlightBonds=list(bond_colors.keys()), highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_path)
    add_title_to_image(output_path, f"Substructure importance")

def custom_cmap():
    colors = [(0.0, "#008bfb"), (0.5, "#FFFFFF"), (1.0, "#ff0051")]
    custom_cmap = LinearSegmentedColormap.from_list("shap_white_center", colors)
    return custom_cmap

# Add titles
def add_title_to_image(image_path, title):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')  # Hide axis
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
