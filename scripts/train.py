#!/usr/bin/env python
"""
Training script for the cVAE model.

This script handles the entire training pipeline including data loading,
model setup, training loop, evaluation, and saving results.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import pyro

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from cvae.data import GEPcached, setup_data_loaders
from cvae.models import CVAE
from cvae.training import ysregularizedELBO
from cvae.utils import compute_decoder_shap_values
from cvae.utils.metrics import get_mae, get_mse, get_yszcorrs
from cvae.training.trainer import run_inference_for_epoch
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro.optim import Adam
from pyro.poutine import block

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_parser():
    parser = argparse.ArgumentParser(description="Train a cVAE model")
    parser.add_argument("--expressionfile", type=str, required=True, help="Path to expression data CSV")
    parser.add_argument("--purityfile", type=str, default=".", help="Path to purity data CSV")
    parser.add_argument("--classfile", type=str, default=".", help="Path to class data CSV")
    parser.add_argument("--outputfolder", type=str, required=True, help="Output folder for results")
    parser.add_argument("--cohort", type=str, required=True, help="Cohort name")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--z_dim", type=int, required=True, help="Latent dimension")
    parser.add_argument("--hidden_layers", type=int, nargs='+', required=True, help="Hidden layer sizes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--annealingfactor", type=float, default=1.0, help="Annealing factor")
    parser.add_argument("--gradregmultiplier", type=float, default=0.0, help="Gradient regularization multiplier")
    parser.add_argument("--corregmultiplier", type=float, default=0.0, help="Correlation regularization multiplier")
    parser.add_argument("--gradregepsilon", type=float, default=1e-8, help="Gradient regularization epsilon")
    parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
    parser.add_argument("--scaleys", action="store_true", help="Scale ys")
    parser.add_argument("--ysmultiplier", type=float, default=1.0, help="ys multiplier")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--sup_num", type=int, default=1000, help="Number of supervised samples")
    parser.add_argument("--val_num", type=int, default=200, help="Number of validation samples")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adamweightdecay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--decodersigma", type=float, default=1.0, help="Decoder sigma")
    parser.add_argument("--reparamepsilon", type=float, default=1e-2, help="Latent std epsilon")
    parser.add_argument("--dropoutp", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--divideannealstart", type=float, default=1.0, help="Divide anneal start")
    parser.add_argument("--shap", action="store_true", help="Compute SHAP values")
    parser.add_argument("--logonly", action="store_true", help="Log only, no output files")
    parser.add_argument("--ysencoder", action="store_true", help="Use ys encoder")
    parser.add_argument("--fulloutput", action="store_true", help="Write full output files")
    return parser

def main(args):
    # Load configuration if provided
    if hasattr(args, "config") and args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    # Set random seed
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ---------- SET FILENAMES ----------
    prefix = "ysVAE_" + args.cohort
    if not args.cuda:
        prefix = "CPU_" + prefix
    if args.batchnorm:
        prefix = prefix + "_batchnorm"
    if args.scaleys:
        prefix = prefix + "_ysscaled"
    if args.ysmultiplier != 1:
        prefix = prefix + "_ysmultiplier" + str(args.ysmultiplier)
    prefix += "_zdim" + str(args.z_dim) + "_numlayers" + str(len(args.hidden_layers))
    prefix += "_epochs" + str(args.num_epochs)
    prefix = prefix + "_anneal" + str(args.annealingfactor)
    prefix = prefix + "_gradregmultiplier" + str(args.gradregmultiplier)
    prefix = prefix + "_corregmultiplier" + str(args.corregmultiplier)
    prefix = prefix + "_gradregeps" + str(args.gradregepsilon)

    logfile = os.path.join(args.outputfolder, "results", prefix + "_logfile.txt")
    savefile = os.path.join(args.outputfolder, "results", prefix + "_model.save")
    outlatentfile = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_latentz.csv")
    outlatentscalefile = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_latentzscale.csv")
    outreconfile = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_inputrecon.csv")
    validation_samples_file = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_validationsamples.csv")
    outpurified = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_cancerGEP.csv")
    outdepurified = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_stromaGEP.csv")

    # ---------- PRE-PROCESS ----------
    std_scaler = StandardScaler()
    TEST_INFILE = args.expressionfile
    test_full = pd.read_csv(TEST_INFILE, index_col=0)
    test_scaled = pd.DataFrame(std_scaler.fit_transform(test_full), columns=test_full.columns)
    test_scaled.index = test_full.index
    data = torch.tensor(test_scaled.values, dtype=torch.float32, device=device)
    genedf = test_scaled
    num_genes = data.size()[1]

    # ------- Class Information -------
    classdf = torch.zeros((data.size(0), 0), dtype=data.dtype, device=device)
    purity_levels = torch.zeros((data.size(0), 1), dtype=data.dtype, device=device)
    cancer_types = torch.zeros((data.size(0), 1), dtype=data.dtype, device=device)
    num_classes = 0
    stratification_classes = None

    if args.purityfile != ".":
        purity_std_scaler = StandardScaler()
        purities_input = pd.read_csv(args.purityfile, index_col=0)
        if args.scaleys:
            purities_scaled = pd.DataFrame(purity_std_scaler.fit_transform(purities_input), columns=purities_input.columns)
            purities_scaled.index = purities_input.index
            purities_test = purities_scaled * args.ysmultiplier
        else:
            purities_test = purities_input * args.ysmultiplier
        purity_levels = torch.tensor(purities_test['purity'].values, dtype=torch.float32, device=device)
        mean_value = torch.nanmean(purity_levels)
        nan_mask = torch.isnan(purity_levels)
        purity_levels[nan_mask] = mean_value
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

    # ---------- SET UP TRAINING ----------
    cvae = CVAE(
        z_dim=args.z_dim,
        input_size=num_genes,
        output_size=num_classes,
        hidden_layers=args.hidden_layers,
        use_cuda=args.cuda,
        annealingfactor=args.annealingfactor,
        decoder_sigma=args.decodersigma,
        decoder_batchnorm=args.batchnorm,
        latentstd=args.reparamepsilon,
        dropoutp=args.dropoutp
    )

    data_loaders = setup_data_loaders(
        dataset=GEPcached, use_cuda=args.cuda, batch_size=args.batch_size,
        sup_num=args.sup_num, val_num=args.val_num,
        input_GEP=genedf,
        input_classes=classdf,
        num_classes=num_classes,
        train_data_size=len(test_scaled),
        stratifyclasses=stratification_classes
    )

    nbatches = args.sup_num // args.batch_size
    nsamples = nbatches * args.batch_size

    indata = GEPcached(
        mode="full", use_cuda=args.cuda,
        sup_num=args.sup_num, val_num=args.val_num,
        train_data_size=len(test_scaled), input_GEP=genedf,
        input_classes=classdf, num_classes=num_classes
    )

    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999), "weight_decay": args.adamweightdecay}
    optimizer = Adam(adam_params)
    milestones = [int(args.num_epochs * 0.5), int(args.num_epochs * 0.75)]
    scheduler = pyro.optim.MultiStepLR({'optimizer': optimizer, 'optim_args': adam_params, 'milestones': milestones, 'gamma': 0.1})

    scaled_model = pyro.poutine.scale(cvae.model, scale=1/(nsamples*(num_genes+num_classes)))
    scaled_guide = pyro.poutine.scale(cvae.guide, scale=1/(nsamples*(num_genes+num_classes)))

    if args.gradregmultiplier == 0 and args.corregmultiplier == 0:
        loss_basic = SVI(scaled_model, scaled_guide, optimizer, loss=Trace_ELBO())
    else:
        loss_basic = SVI(
            scaled_model, scaled_guide, optimizer,
            loss=ysregularizedELBO(
                grad_reg_multiplier=args.gradregmultiplier,
                cor_reg_multiplier=args.corregmultiplier,
                grad_reg_epsilon=args.gradregepsilon,
            )
        )
    losses = [loss_basic]

    # Additional losses for recon and KL
    model_no_obs2 = block(scaled_model, hide=["z"])
    guide_no_y2 = block(scaled_guide, hide=["z"])
    optim3 = Adam({"lr": 0.001})
    reconloss = SVI(model_no_obs2, guide_no_y2, optim3, loss=TraceMeanField_ELBO())
    losses.append(reconloss)

    model_no_obs = block(scaled_model, hide=["obs"])
    optim2 = Adam({"lr": 0.001})
    KLloss = SVI(model_no_obs, scaled_guide, optim2, loss=TraceMeanField_ELBO())
    losses.append(KLloss)

    # ---------- TRAIN MODEL ----------
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        os.makedirs(os.path.dirname(outlatentfile), exist_ok=True)
        logger = open(logfile, "w")
        pyro.clear_param_store()

        if args.gradregmultiplier == 0 and args.corregmultiplier == 0:
            print("loss is ELBO only")
            columns = "epoch,loss_ELBO,loss_recon,loss_KLdiv,mae_train,mae_test,mse_train,mse_test,dcor_train,dcor_test"
        else:
            columns = "epoch,loss_total,loss_recon,loss_KLdiv,loss_ELBO,loss_grad,loss_dcor,mae_train,mae_test,mse_train,mse_test,dcor_train,dcor_test"
        logger.write(columns.replace(",", "\t") + "\n")

        cvae.annealingfactor = args.annealingfactor / args.divideannealstart
        annealing_factor_increment = args.annealingfactor / (args.num_epochs / 2)

        for i in range(args.num_epochs):
            cvae.annealingfactor = min(args.annealingfactor, cvae.annealingfactor + annealing_factor_increment)

            if args.gradregmultiplier == 0 and args.corregmultiplier == 0:
                epoch_losses_sup = run_inference_for_epoch(data_loaders, losses, args.cuda)
            else:
                import io
                from contextlib import redirect_stdout
                losssums = [0, 0, 0]
                strout = io.StringIO()
                with redirect_stdout(strout):
                    epoch_losses_sup = run_inference_for_epoch(data_loaders, losses, args.cuda)
                for inner_list in [x.split(',') for x in strout.getvalue().split('\n') if x]:
                    for idx, value in enumerate(inner_list):
                        losssums[idx] += float(value)
                elbo_loss = losssums[0] / nbatches
                reg_loss = losssums[1] / nbatches
                cor_loss = losssums[2] / nbatches

            scheduler.step()
            avg_epoch_losses_sup = [f"{(value/nbatches):.6f}" for value in epoch_losses_sup]
            str_print = str(i)
            str_loss_sup = "\t".join(avg_epoch_losses_sup)
            str_print += "\t" + str_loss_sup

            if args.gradregmultiplier != 0 and args.corregmultiplier != 0:
                str_print += f"\t{elbo_loss:0.6f}\t{reg_loss:0.6f}\t{cor_loss:0.6f}"

            train_mae = get_mae(data_loaders["test"], cvae.recon_fn, args.batch_size)
            str_print += f"\t{train_mae:0.9f}"
            test_mae = get_mae(data_loaders["valid"], cvae.recon_fn, args.batch_size)
            str_print += f"\t{test_mae:0.9f}"
            train_mse = get_mse(data_loaders["test"], cvae.recon_fn, args.batch_size)
            str_print += f"\t{train_mse:0.9f}"
            test_mse = get_mse(data_loaders["valid"], cvae.recon_fn, args.batch_size)
            str_print += f"\t{test_mse:0.9f}"
            train_corr = get_yszcorrs(data_loaders["test"], cvae.get_z, None, cor_reg_multiplier=1)
            str_print += f"\t{train_corr:0.9f}"
            test_corr = get_yszcorrs(data_loaders["valid"], cvae.get_z, None, cor_reg_multiplier=1)
            str_print += f"\t{test_corr:0.9f}"

            logger.write(str_print + "\n")
            logger.flush()

    finally:
        logger.close()
        pyro.get_param_store().save(savefile)
        test_scaled.to_csv(os.path.join(args.outputfolder, "results", "encoded_datapoints", f"{str(args.cohort).split('_')[0]}_{num_genes}genes_logCPM_zscaled.csv"))

        if args.shap:
            xs_bg, ys_bg = next(iter(data_loaders["test"]))
            with torch.no_grad():
                z_bg, _ = cvae.get_z(xs_bg, ys_bg)
            background_inputs = (z_bg, ys_bg)
            xs_target, ys_target = next(iter(data_loaders["valid"]))
            with torch.no_grad():
                z_target, _ = cvae.get_z(xs_target, ys_target)
            target_inputs = (z_target, ys_target)
            shap_values = compute_decoder_shap_values(cvae, background_inputs, target_inputs)
            outshapfile = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_decoder_shap.csv")
            pd.DataFrame(shap_values[0]).to_csv(outshapfile)

        if not args.logonly:
            argparse_out = os.path.join(args.outputfolder, "results", "encoded_datapoints", prefix + "_argparse.csv")
            with open(argparse_out, 'w') as f:
                for arg in vars(args):
                    f.write(f"{arg} : {getattr(args, arg)}\n")

            cvae.eval()
            with torch.no_grad():
                if args.ysencoder:
                    encoded_z = cvae.encoder_z([indata.data, indata.targets])
                else:
                    encoded_z = cvae.encoder_z(indata.data)
                decoded_zmeans = cvae.decoder([encoded_z[0], indata.targets])

            pd.DataFrame(encoded_z[0].cpu().detach().numpy(), index=test_scaled.index.values).to_csv(outlatentfile)
            pd.DataFrame(encoded_z[1].cpu().detach().numpy(), index=test_scaled.index.values).to_csv(outlatentscalefile)
            pd.DataFrame(decoded_zmeans.cpu().detach().numpy(), index=test_scaled.index.values).to_csv(outreconfile)
            pd.DataFrame(data_loaders['valid'].dataset.valid_indices).to_csv(validation_samples_file)

        print("Run finished : " + prefix)

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args)