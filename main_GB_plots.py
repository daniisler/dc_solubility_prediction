from rdkit.Chem import Descriptors
from rdkit import Chem

from GB_plot_functions import train_GB_model, make_plots

best_params = {
    "num_leaves": 433,
    "learning_rate": 0.045151338822694405,
    "n_estimators": 1779,
    "max_depth": 14,
    "subsample": 0.9378226093192981,
    "colsample_bytree": 0.5739475094314461
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

solvents = []

train_GB_model(
    input_data_filepath='input_data/AqSolDB_filtered_log.csv',
    output_data_filepath='saved_models/gradient_boosting/Aq_ap_fp_only_opt_lr/model',
    best_params=best_params,
    selected_fp=selected_fp,
    descriptors=descriptors,
    solvents=solvents,
    group_kfold=True
)

make_plots('saved_models/gradient_boosting/Aq_ap_fp_only_opt_lr/model.pkl',
           saving_dir='C:/Users/david/OneDrive - ETH Zurich/ETH/Informatik/Digital Chemistry/Project/Plots/Aq_ap_fp',
           saving_name='AqSolDB_aq_fp')

