import os
from pickle import load
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from nn_model import SolubilityModel
from logger import logger

logger = logger.getChild("predict")


def predict_solubility_from_smiles(
    smiles,
    model_save_dir,
    best_hyperparams,
    T=None,
    solvent="water",
    selected_fp=None,
    use_rdkit_descriptors=False,
    descriptors_list=None,
    missing_rdkit_desc=0.0,
    solvent_fp=False,
    solvent_smiles="",
    scale_transform=True,
):
    """Predict the solubility of a molecule given its SMILES representation.

    :param str smiles: SMILES representation of the molecule
    :param str model_save_dir: directory where the trained models are saved
    :param dict best_hyperparams: best hyperparameters for the models
    :param float T: temperature used for prediction
    :param str solvent: solvent used for prediction; None for all solvents
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys: 'm_fp', 'rd_fp', 'ap_fp', 'tt_fp'
    :param bool use_rdkit_descriptors: whether to include rdkit descriptors
    :param list descriptors_list: list of rdkit descriptors to use; None for all descriptors
    :param float missing_rdkit_desc: missing value replacement for the rdkit descriptors
    :param bool solvent_fp: whether to include solvent fingerprints
    :param str solvent_smiles: SMILES representation of the solvent
    :param bool scale_transform: whether to scale the input data

    :return: predicted solubility (float)

    """
    # Set the default object-kind input parameters
    if selected_fp is None:
        selected_fp = {"m_fp": (2048, 2)}
    if descriptors_list is None:
        descriptors_list = ["all"]

    # Load the trained model
    # model = torch.load(os.path.join(model_save_dir, 'architecture.pth'))
    model = SolubilityModel(
        train_data=None,
        valid_data=None,
        test_data=None,
        input_size=sum(selected_fp[key][0] for key in selected_fp.keys()),
        n_neurons_hidden_layers=best_hyperparams["n_neurons_hidden_layers"],
        activation_function=best_hyperparams["activation_fn"],
        loss_function=best_hyperparams["loss_fn"],
        optimizer=best_hyperparams["optimizer"],
        lr=best_hyperparams["learning_rate"],
        batch_size=best_hyperparams["batch_size"],
    )

    model.load_state_dict(
        torch.load(os.path.join(model_save_dir, f"weights_{solvent}.pth"))
    )
    model.eval()

    # Calculate the fingerprints
    mol = MolFromSmiles(smiles)
    if solvent_fp:
        mol_solvent = MolFromSmiles(solvent_smiles)
    X = []
    for key in selected_fp.keys():
        if key == "m_fp":
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                fpSize=selected_fp["m_fp"][0], radius=selected_fp["m_fp"][1]
            )
            X.append(
                torch.tensor(
                    np.array(mfpgen.GetFingerprint(mol)), dtype=torch.float32
                ).reshape(1, -1)
            )
            if solvent_fp:
                X.append(
                    torch.tensor(
                        np.array(mfpgen.GetFingerprint(mol)), dtype=torch.float32
                    ).reshape(1, -1)
                )
        if key == "rd_fp":
            rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=selected_fp["rd_fp"][0],
                minPath=selected_fp["rd_fp"][1][0],
                maxPath=selected_fp["rd_fp"][1][1],
            )
            X.append(
                torch.tensor(
                    np.array(rdkgen.GetFingerprint(mol)), dtype=torch.float32
                ).reshape(1, -1)
            )
            if solvent_fp:
                X.append(
                    torch.tensor(
                        np.array(rdkgen.GetFingerprint(mol)), dtype=torch.float32
                    ).reshape(1, -1)
                )
        if key == "ap_fp":
            apgen = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=selected_fp["ap_fp"][0],
                minDistance=selected_fp["ap_fp"][1][0],
                maxDistance=selected_fp["ap_fp"][1][1],
            )
            X.append(
                torch.tensor(
                    np.array(apgen.GetFingerprint(mol)), dtype=torch.float32
                ).reshape(1, -1)
            )
            if solvent_fp:
                X.append(
                    torch.tensor(
                        np.array(apgen.GetFingerprint(mol)), dtype=torch.float32
                    ).reshape(1, -1)
                )
        if key == "tt_fp":
            ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                fpSize=selected_fp["tt_fp"][0], torsionAtomCount=selected_fp["tt_fp"][1]
            )
            X.append(
                torch.tensor(
                    np.array(ttgen.GetFingerprint(mol)), dtype=torch.float32
                ).reshape(1, -1)
            )
            if solvent_fp:
                X.append(
                    torch.tensor(
                        np.array(ttgen.GetFingerprint(mol)), dtype=torch.float32
                    ).reshape(1, -1)
                )

    if use_rdkit_descriptors:
        # TODO: Implement the descriptor calculation for the molecule
        pass

    X = np.concatenate(X, axis=1).reshape(1, -1)
    # Scale the input data according to the saved scaler
    if scale_transform:
        with open(os.path.join(model_save_dir, f"scaler_{solvent}.pkl"), "rb") as f:
            scaler = load(f)
        X = scaler.transform(X)

    # Predict the solubility
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        solubility = model(X).item()

    return solubility
