import os

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, rdFingerprintGenerator


def predict_solubility_from_smiles(
    smiles,
    model_save_dir,
    selected_fp=None,
    descriptors=None,
    solvent_smiles="",
    scale_transform=True,
    T=298.15,
):
    """Predict the solubility of a molecule given its SMILES representation.

    :param str smiles: SMILES representation of the molecule
    :param str model_save_dir: directory where the trained models are saved
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys: 'm_fp', 'rd_fp', 'ap_fp', 'tt_fp'
    :param float missing_rdkit_desc: missing value replacement for the rdkit descriptors
    :param bool solvent_fp: whether to include solvent fingerprints
    :param str solvent_smiles: SMILES representation of the solvent
    :param bool scale_transform: whether to scale the input data
    :param float T: temperature in K

    :return: predicted solubility (float)

    """
    # Set the default object-kind input parameters
    if selected_fp is None:
        selected_fp = {"m_fp": (2048, 2)}

    # Load model
    model = joblib.load(os.path.join(model_save_dir, f"model_GB.pkl"))

    # Calculate the fingerprints
    mol = MolFromSmiles(smiles)
    mol_solvent = MolFromSmiles(solvent_smiles)
    X = []
    X_sol = []
    for key in selected_fp.keys():
        if key == "m_fp":
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                fpSize=selected_fp["m_fp"][0], radius=selected_fp["m_fp"][1]
            )
            X.append(np.array(mfpgen.GetFingerprint(mol)).reshape(1, -1))
            X_sol.append(np.array(mfpgen.GetFingerprint(mol_solvent)).reshape(1, -1))
        if key == "rd_fp":
            rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=selected_fp["rd_fp"][0],
                minPath=selected_fp["rd_fp"][1][0],
                maxPath=selected_fp["rd_fp"][1][1],
            )
            X.append(np.array(rdkgen.GetFingerprint(mol)).reshape(1, -1))
            X_sol.append(np.array(rdkgen.GetFingerprint(mol_solvent)).reshape(1, -1))
        if key == "ap_fp":
            apgen = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=selected_fp["ap_fp"][0],
                minDistance=selected_fp["ap_fp"][1][0],
                maxDistance=selected_fp["ap_fp"][1][1],
            )
            X.append(np.array(apgen.GetFingerprint(mol)).reshape(1, -1))
            X_sol.append(np.array(apgen.GetFingerprint(mol_solvent)).reshape(1, -1))
        if key == "tt_fp":
            ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                fpSize=selected_fp["tt_fp"][0], torsionAtomCount=selected_fp["tt_fp"][1]
            )
            X.append(np.array(ttgen.GetFingerprint(mol)).reshape(1, -1))
            X_sol.append(np.array(ttgen.GetFingerprint(mol_solvent)).reshape(1, -1))
    # As in the training script, the solute fingerprints are all before the solvent fingerprints
    X += X_sol
    if descriptors is not None:
        for desc_name, desc_func in descriptors.items():
            X.append([np.array([desc_func(mol)])])
            X.append([np.array([desc_func(mol_solvent)])])
    X.append(np.array([T]).reshape(1, -1))
    X = np.concatenate(X, axis=1).reshape(1, -1)
    # Scale the input data according to the saved scaler
    if scale_transform:
        scaler = joblib.load(os.path.join(model_save_dir, f"scaler_GB.pkl"))
        X = scaler.transform(X)

    # Predict the solubility
    solubility = model.predict(X)

    return solubility

# Example usage
if __name__ == "__main__":
    selected_fp = {'ap_fp': (2048, (1, 30))}
    descriptors = {
        'MolLogP': Descriptors.MolLogP,
        'LabuteASA': Descriptors.LabuteASA,
        'MolWt': Descriptors.MolWt,
        'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,
        'Kappa3': Descriptors.Kappa3,
        'PEOE_VSA2': Descriptors.PEOE_VSA2,
        'PEOE_VSA9': Descriptors.PEOE_VSA9,
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
    scale_transform = False
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    smiles = "CCO"
    solvent_smiles = "O"
    T = 298.15
    solubility = predict_solubility_from_smiles(
        smiles, model_save_dir, selected_fp, descriptors, solvent_smiles, scale_transform
    )
    print(f"Predicted solubility: logS={solubility} for SMILES: {smiles} in solvent: {solvent_smiles} at T={T} K")