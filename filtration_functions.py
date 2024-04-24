import pandas as pd
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