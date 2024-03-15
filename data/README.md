## Plasmodium falciparum 3D7 Data

The data(`01_raw/plasmodium_falciparum_3d7_assays.csv`) contains assays related to Plasmodium falciparum 3D7 strains. It  was retrieved from the CheMBL database(version 33).

### Data Cleaning
The code used for data cleaning is in the notebook named `raw_data_cleaning.ipynb`.

The `standard_value` was converted to microMolar(uM) units for consistency.
Only records having a non-null pchembl_values were retained. 

### IC50 dataset
The final dataset(**2082 records**) consists of IC50 values(in uM) for various compounds tested against the parasite.
- **File:** `plasmodium_falciparum_3d7_ic50.csv`
- **Columns:** 
    - `smiles`: Canonical SMILES representation of the compound.
    - `ic50`: Standard value measured in microMolar (uM) units, after conversion.

