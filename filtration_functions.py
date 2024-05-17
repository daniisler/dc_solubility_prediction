import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Lipinski

df = pd.read_csv('input_data/BigSolDB.csv')


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


def filter_solvent(df, solvents, get_same_solvent=True):
    """Filter given dataframe based on given solvent (get rows with given solvent or rows with different solvent)

    :param dataframe df: dataframe of input data
    :param str solvent: solvent used for filtering
    :param bool get_same_solvent: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    for solvent in solvents:
        if get_same_solvent:
            df = df[df['Solvent'] == solvent]
            if df.empty:
                ValueError('No entries in data with given solvent as solvent')
    return df
    #return df[df['Solvent'] != solvent]


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
            return df  # not sure if return 0 would be better
        return df
    return df[df['SMILES_Solvent'] != smiles_solvent]


def filter_molecule_substructure(df, smiles_substructures, is_substructure_in_molecule=True):
    """Filter molecules based on given list of smiles of substructures (get rows where given substructures are part of
    the molecule, if is_substructure_in_molecule is True)

    :param dataframe df: dataframe of input data
    :param list of strings smiles_substructures: smiles of substructure(s) used for filtering
    :param bool is_substructure_in_molecule: True if values rows with given molecule should be returned, false otherwise
    :return: filtered dataframe
    """
    mol_substructures = []
    for smiles_sub in smiles_substructures:
        mol_substructures.append(Chem.MolFromSmiles(smiles_sub))
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES',
                                         molCol='mol_molecule')  # should be removed if df already has molecule column
    if is_substructure_in_molecule:
        for mol_sub in mol_substructures:
            mask = []
            for mol_mol in df['mol_molecule']:
                mask.append(mol_mol.HasSubstructMatch(mol_sub))
            df = df[mask]
    else:
        for mol_sub in mol_substructures:
            mask = []
            for mol_mol in df['mol_molecule']:
                mask.append(mol_mol.HasSubstructMatch(mol_sub))
            df = df[[not elem for elem in mask]]  # round about way to flip booleans in mask
    return df.drop(['mol_molecule'], axis=1)


def filter_solvent_substructure(df, smiles_substructures, is_substructure_in_solvent=True):
    """Filter solvents based on given list of smiles of substructures (get rows where given substructures are part of
    the solvent, if is_substructure_in_solvent is True)

    :param dataframe df: dataframe of input data
    :param list of strings smiles_substructures: smiles of substructure(s) used for filtering
    :param bool is_substructure_in_solvent: True if values rows with given solvent should be returned, false otherwise
    :return: filtered dataframe
    """
    mol_substructures = []
    for smiles_sub in smiles_substructures:
        mol_substructures.append(Chem.MolFromSmiles(smiles_sub))
    df = df[~(df['SMILES_Solvent'] == '-')]  # Remove rows without smiles for solvent
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES_Solvent', molCol='mol_solvent')
    if is_substructure_in_solvent:
        for mol_sub in mol_substructures:
            mask = []
            for mol_sol in df['mol_solvent']:
                mask.append(mol_sol.HasSubstructMatch(mol_sub))
            df = df[mask]
    else:
        for mol_sub in mol_substructures:
            mask = []
            for mol_sol in df['mol_solvent']:
                mask.append(mol_sol.HasSubstructMatch(mol_sub))
            df = df[[not elem for elem in mask]]  # round about way to flip booleans in mask
    return df.drop(['mol_solvent'], axis=1)


def filter_molecule_heteroatoms(df, heteroatoms_in_molecule=False):
    """Filter molecules based on whether they contain heteroatoms or not.

    :param dataframe df: dataframe of input data
    :param bool heteroatoms_in_molecule: False if molecule with heteroatoms should be removed, True otherwise
    :return: filtered dataframe
    """
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES', molCol='mol_molecule')  # should be removed if df already has molecule column
    num_heteroatoms = df['mol_molecule'].apply(Lipinski.NumHeteroatoms)
    mask = list(map(bool, num_heteroatoms))
    if heteroatoms_in_molecule:
        return df[mask].drop(['mol_molecule'], axis=1)
    return df[[not elem for elem in mask]].drop(['mol_molecule'], axis=1)


def filter_solvent_heteroatoms(df, heteroatoms_in_solvent=False):
    """Filter solvents based on whether they contain heteroatoms or not.

    :param dataframe df: dataframe of input data
    :param bool heteroatoms_in_solvent: False if solvents with heteroatoms should be removed, True otherwise
    :return: filtered dataframe
    """
    df = df[~(df['SMILES_Solvent'] == '-')]  # Remove rows without smiles for solvent
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES_Solvent', molCol='mol_solvent')  # should be removed if df already has molecule column
    num_heteroatoms = df['mol_solvent'].apply(Lipinski.NumHeteroatoms)
    mask = list(map(bool, num_heteroatoms))
    if heteroatoms_in_solvent:
        return df[mask].drop(['mol_solvent'], axis=1)
    return df[[not elem for elem in mask]].drop(['mol_solvent'], axis=1)


def filter_solvent_h_bonds(df, h_acceptor, h_donor):
    """Filter solvents based on whether they are H donors or/and H acceptors.

    :param dataframe df: dataframe of input data
    :param bool h_acceptor: True if solvent should be H acceptor, False otherwise
    :param bool h_donor: True if solvent should be H donor, False otherwise
    :return: filtered dataframe
    """
    df = df[~(df['SMILES_Solvent'] == '-')]  # Remove rows without smiles for solvent
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES_Solvent', molCol='mol_solvent')
    num_h_acceptor = df['mol_solvent'].apply(Lipinski.NumHAcceptors)
    num_h_donor = df['mol_solvent'].apply(Lipinski.NumHDonors)
    mask_h_acceptor = list(map(bool, num_h_acceptor))
    mask_h_donor = list(map(bool, num_h_donor))
    for index, row in df.iterrows():  # for some reason NumHAcceptors and NumHDonors says water is neither an acceptor or donor
        if row['Solvent'] == 'water':  # change value for water to true for both
            mask_h_acceptor[index] = True
            mask_h_donor[index] = True
    if h_acceptor:
        if h_donor:
            mask = np.logical_and(mask_h_donor, mask_h_acceptor)  # only True if True in both masks
            return df[mask].drop(['mol_solvent'], axis=1)
        return df[mask_h_acceptor].drop(['mol_solvent'], axis=1)
    if h_donor:
        return df[mask_h_donor].drop(['mol_solvent'], axis=1)
    mask = np.logical_or(mask_h_donor, mask_h_acceptor)  # True if True in either of masks
    return df[[not elem for elem in mask]].drop(['mol_solvent'], axis=1)
