from rdkit.Chem import Descriptors
from rdkit import Chem

from GB_plot_functions import train_GB_model, make_plots

best_params = {
    "num_leaves": 114,
    "learning_rate": 0.009279770914952072,
    "n_estimators": 1691,
    "max_depth": 29,
    "subsample": 0.7578799877302553,
    "colsample_bytree": 0.5004152458974538
}

selected_fp = {'ap_fp': (2048, (1, 30))}

descriptors = {
    'molecular_weight': Descriptors.MolWt,
    'TPSA': Descriptors.TPSA,
    'num_h_donors': Descriptors.NumHDonors,
    'num_h_acceptors': Descriptors.NumHAcceptors,
    'num_rotatable_bonds': Descriptors.NumRotatableBonds,
    'num_atoms': Chem.rdchem.Mol.GetNumAtoms,
    'num_heteroatoms': Descriptors.NumHeteroatoms,
    'num_valence_electrons': Descriptors.NumValenceElectrons,
    'num_rings': Descriptors.RingCount,
    'max_abs_partial_charge': Descriptors.MaxAbsPartialCharge,
    'max_partial_charge': Descriptors.MaxPartialCharge,
    'min_abs_partial_charge': Descriptors.MinAbsPartialCharge,
    'min_partial_charge': Descriptors.MinPartialCharge,
    'num_NHOH': Descriptors.NHOHCount,
    'fraction_C_sp3': Descriptors.FractionCSP3
}


# train_GB_model(
#     input_data_filepath='input_data/BigSolDB_filtered_log.csv',
#     output_data_filepath='saved_models/gradient_boosting/Big_all_in/model',
#     best_params=best_params,
#     selected_fp=selected_fp,
#     descriptors=descriptors,
#     solvents=[],
#     group_kfold=True
# )

make_plots('saved_models/gradient_boosting/Big_all_in/model.pkl',
           saving_dir='C:/Users/david/OneDrive - ETH Zurich/ETH/Informatik/Digital Chemistry/Project/Plots',
           saving_name='Big_all_in')