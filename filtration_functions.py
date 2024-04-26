import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

def filter_weight(df, weight, get_larger_values=True):
    """Filter given dataframe based on given weight (get rows with larger/smaller weights)

    :param dataframe df: dataframe of input data
    :param float weight: weight value used for filtering
    :param bool get_larger_values: True if values larger than weight should be returned, false otherwise
    :return: filtered dataframe
    """
    if get_larger_values:
        return df[df['weight'] >= weight]
    return df[df['weight'] <= weight]

def filter_solubility(df, solubility, get_larger_values=True):
    """Filter given dataframe based on given solubility (get rows with larger/smaller solubility)

    :param dataframe df: dataframe of input data
    :param float solubility: solubility value used for filtering
    :param bool get_larger_values: True if values larger than weight should be returned, false otherwise
    :return: filtered dataframe
    """
    if get_larger_values:
        return df[df['Solubility'] >= solubility]
    return df[df['Solubility'] <= solubility]

def filter_temperature(df, temperature, get_larger_values=True):
    """Filter given dataframe based on given solubility (get rows with larger/smaller solubility)

    :param dataframe df: dataframe of input data
    :param float temperature: solubility value used for filtering
    :param bool get_larger_values: True if values larger than weight should be returned, false otherwise
    :return: filtered dataframe
    """
    if get_larger_values:
        return df[df['T,K'] >= temperature]
    return df[df['T,K'] <= temperature]

def filter_solvent(df, solvent, get_same_solvent=True):
    """Filter given dataframe based on given solvent (get rows with given solvent or rows with different solvent)

    :param dataframe df: dataframe of input data
    :param str solvent: solvent used for filtering
    :param bool get_same_solvent: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    if get_same_solvent:
        df = df[df['Solvent'] == solvent]
        if df.empty:
            print('Solvent not found in database')
            return df # not sure if return 0 would be better
        return df
    return df[df['Solvent'] != solvent]

def filter_solvent_smiles(df, smiles_solvent, get_same_solvent=True):
    """Filter given dataframe based on given smiles of solvent (get rows with given solvent or rows with different solvent)

    :param dataframe df: dataframe of input data
    :param str smiles_solvent: smiles of solvent used for filtering
    :param bool get_same_solvent: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    if get_same_solvent:
        df = df[df['SMILES_Solvent'] == smiles_solvent]
        if df.empty:
            print('Solvent not found in database')
            return df # not sure if return 0 would be better
        return df
    return df[df['SMILES_Solvent'] != smiles_solvent]

def filter_molecule_substructure(df, smiles_substructure, is_substructure_in_molecule=True):
    """Filter given dataframe based on given smiles of substructure (get rows where given substructure is part of the molecule,
    if is_substructure_in_molecule is True)

    :param dataframe df: dataframe of input data
    :param str smiles_substructure: smiles of substructure used for filtering
    :param bool is_substructure_in_molecule: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    mol_substructure = Chem.MolFromSmiles(smiles_substructure)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES', molCol='mol_molecule') # should be removed if df already has molecule column
    mask = []
    for mol in df['mol_molecule']:
        mask.append(mol.HasSubstructMatch(mol_substructure))
    df = df.drop(['mol_molecule'], axis=1) # should be removed if df already has molecule column
    if is_substructure_in_molecule:
        return df[mask]
    return df[[not elem for elem in mask]] # round about way to flip booleans in mask

# TODO: get rid of SMILES Parse Error
def filter_solvent_substructure(df, smiles_substructure, is_substructure_in_solvent=True):
    """Filter given dataframe based on given smiles of substructure (get rows where given substructure is part of the solvent,
    if is_substructure_in_molecule is True)

    :param dataframe df: dataframe of input data
    :param str smiles_substructure: smiles of substructure used for filtering
    :param bool is_substructure_in_solvent: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    mol_substructure = Chem.MolFromSmiles(smiles_substructure)
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES_Solvent', molCol='mol_solvent') # should be removed if df already has molecule column
    mask = []
    for mol_sol in df['mol_solvent']:
        mask.append(mol_sol.HasSubstructMatch(mol_substructure))
    df = df.drop(['mol_solvent'], axis=1) # should be removed if df already has molecule column
    if is_substructure_in_solvent:
        return df[mask]
    return df[[not elem for elem in mask]] # round about way to flip booleans in mask

