#!/usr/bin/env python
"""
Evaluation script for the cVAE model.

This script loads a trained cVAE model, evaluates it on provided data,
computes metrics, and saves results/visualizations.
"""

import argparse
import os
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from cvae.data import GEPcached, setup_data_loaders
from cvae.models import CVAE
from cvae.utils.metrics import get_mae, get_mse, get_yszcorrs

def create_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained cVAE model")
    parser.add_argument("--model_file", type=str, required=True, help="Path to saved model parameters (.save)")
    parser.add_argument("--expressionfile", type=str, required=True, help="Path to expression data CSV")
    parser.add_argument("--purityfile", type=str, default=".", help="Path to purity data CSV")
    parser.add_argument("--classfile", type=str, default=".", help="Path to class data CSV")
    parser.add_argument("--outputfolder", type=str, required=True, help="Output folder for evaluation results")
    parser.add_argument("--z_dim", type=int, required=True, help="Latent dimension")
    parser.add_argument("--hidden_layers", type=int, nargs='+', required=True, help="Hidden layer sizes")
    parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
    parser.add_argument("--decodersigma", type=float, default=1.0, help="Decoder sigma")
    parser.add_argument("--reparamepsilon", type=float, default=1e-2, help="Latent std epsilon")
    parser.add_argument("--dropoutp", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser

def main(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load and preprocess data
    std_scaler = StandardScaler()
    test_full = pd.read_csv(args.expressionfile, index_col=0)
    test_scaled = pd.DataFrame(std_scaler.fit_transform(test_full), columns=test_full.columns)
    test_scaled.index = test_full.index
    data = torch.tensor(test_scaled.values, dtype=torch.float32, device=device)
    genedf = test_scaled
    num_genes = data.size()[1]

    classdf = torch.zeros((data.size(0), 0), dtype=data.dtype, device=device)
    num_classes = 0
    stratification_classes = None

    if args.purityfile != ".":
        purity_std_scaler = StandardScaler()
        purities_input = pd.read_csv(args.purityfile, index_col=0)
        purities_scaled = pd.DataFrame(purity_std_scaler.fit_transform(purities_input), columns=purities_input.columns)
        purities_scaled.index = purities_input.index
        purity_levels = torch.tensor(purities_scaled['purity'].values, dtype=torch.float32, device=device)
        purity_levels = purity_levels.unsqueeze(1)
        classdf = torch.cat([classdf, purity_levels], dim=1)

    if args.classfile != ".":
        ohe = OneHotEncoder(sparse_output=False)
        cancer_classes = pd.read_csv(args.classfile, index_col=0)
        encoded_disease = ohe.fit_transform(cancer_classes)
        cancer_types = torch.tensor(encoded_disease, dtype=torch.float32, device=device)
        classdf = torch.cat([classdf, cancer_types], dim=1)
        stratification_classes = encoded_disease[:, [0, 1]]

    non_zero_columns = (classdf != 0).any(dim=0)
    num_classes = classdf[:, non_zero_columns].size(1) if classdf.size(1) > 0 else 0

    # Instantiate model
    cvae = CVAE(
        z_dim=args.z_dim,
        input_size=num_genes,
        output_size=num_classes,
        hidden_layers=args.hidden_layers,
        use_cuda=args.cuda,
        annealingfactor=1.0,
        decoder_sigma=args.decodersigma,
        decoder_batchnorm=args.batchnorm,
        latentstd=args.reparamepsilon,
        dropoutp=args.dropoutp
    )

    # Load model parameters
    cvae.to(device)
    pyro_param_store = torch.load(args.model_file, map_location=device)
    cvae.load_state_dict(pyro_param_store, strict=False)

    # Set up data loaders
    data_loaders = setup_data_loaders(
        dataset=GEPcached, use_cuda=args.cuda, batch_size=args.batch_size,
        sup_num=len(test_scaled), val_num=0,
        input_GEP=genedf,
        input_classes=classdf,
        num_classes=num_classes,
        train_data_size=len(test_scaled),
        stratifyclasses=stratification_classes
    )

    # Evaluate
    cvae.eval()
    with torch.no_grad():
        mae = get_mae(data_loaders["test"], cvae.recon_fn, args.batch_size)
        mse = get_mse(data_loaders["test"], cvae.recon_fn, args.batch_size)
        corr = get_yszcorrs(data_loaders["test"], cvae.get_z, None, cor_reg_multiplier=1)

    os.makedirs(args.outputfolder, exist_ok=True)
    results_file = os.path.join(args.outputfolder, "evaluation_metrics.txt")
    with open(results_file, "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"Correlation: {corr}\n")

    print(f"Evaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)