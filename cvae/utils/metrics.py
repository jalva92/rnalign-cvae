"""
Evaluation metrics and analysis utilities for cVAE models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Callable, Tuple, Optional, Any, List

def compute_au(model: Any, test_data_batch: Any, delta: float = 0.01) -> Tuple[int, torch.Tensor]:
    """
    Compute the number of active units in the latent space of a VAE.

    Args:
        model: The VAE model with an encode_stats method.
        test_data_batch: Iterable of batches (torch.Tensor) for evaluation.
        delta: Threshold for considering a unit as active.

    Returns:
        Tuple containing:
            - Number of active units (int)
            - Variance of means for all latent variables (torch.Tensor)
    """
    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)
    au_var = var_sum / (cnt - 1)
    return (au_var >= delta).sum().item(), au_var

def get_mae(
    data_loader: Any,
    recon_fn: Callable,
    batch_size: int
) -> float:
    """
    Compute mean absolute error (MAE) over a data loader.

    Args:
        data_loader: Iterable yielding (xs, ys) batches.
        recon_fn: Function to reconstruct xs from xs, ys.
        batch_size: Batch size used for normalization.

    Returns:
        MAE value (float).
    """
    predictions, actuals = [], []
    mae = 0
    get_mae_loss = nn.L1Loss()
    for xs, ys in data_loader:
        predictions.append(recon_fn(xs, ys))
        actuals.append(xs)
    for pred, act in zip(predictions, actuals):
        mae += get_mae_loss(act, pred)
    set_mae = mae / (len(predictions) * batch_size)
    return set_mae

def get_mse(
    data_loader: Any,
    recon_fn: Callable,
    batch_size: int
) -> float:
    """
    Compute mean squared error (MSE) over a data loader.

    Args:
        data_loader: Iterable yielding (xs, ys) batches.
        recon_fn: Function to reconstruct xs from xs, ys.
        batch_size: Batch size used for normalization.

    Returns:
        MSE value (float).
    """
    predictions, actuals = [], []
    mse = 0
    get_mse_loss = nn.MSELoss()
    for xs, ys in data_loader:
        predictions.append(recon_fn(xs, ys))
        actuals.append(xs)
    for pred, act in zip(predictions, actuals):
        mse += get_mse_loss(act, pred)
    set_mse = mse / (len(predictions) * batch_size)
    return set_mse

def get_yszcorrs(
    data_loader: Any,
    get_z: Callable,
    corr_fn: Callable,
    cor_reg_multiplier: float = 1
) -> float:
    """
    Compute normalized correlation between ys and zs for each batch.

    Args:
        data_loader: Iterable yielding (xs, ys) batches.
        get_z: Function to get zmeans from xs, ys.
        corr_fn: Correlation function.
        cor_reg_multiplier: Multiplier for regularization (default 1).

    Returns:
        Normalized correlation (float).
    """
    norm_corrs = []
    for xs, ys in data_loader:
        zmeans, _ = get_z(xs, ys)
        norm_corrs.append(corr_fn(zmeans, ys) / zmeans.shape[1] / zmeans.shape[0])
    normalised_correlation = sum(norm_corrs) / len(norm_corrs)
    return normalised_correlation

def compare_VAE_LM(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    gene_names: List[str],
    test_samples: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compare VAE and linear model outputs for each gene using MSE, MAE, and Levene's test.

    Args:
        tensor1: First tensor (samples x features).
        tensor2: Second tensor (samples x features).
        gene_names: List of gene names (length = features).
        test_samples: Optional list of sample indices to use.

    Returns:
        DataFrame with MSE, MAE, Levene_Stat, and P_Value for each gene.
    """
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape."
    assert len(gene_names) == tensor1.shape[1], "The length of gene_names must equal the number of features (columns)."
    results = []
    for feature_idx in range(tensor1.shape[1]):
        if test_samples:
            feature_data1 = tensor1[test_samples, feature_idx].detach().numpy()
            feature_data2 = tensor2[test_samples, feature_idx].detach().numpy()
        else:
            feature_data1 = tensor1[:, feature_idx].detach().numpy()
            feature_data2 = tensor2[:, feature_idx].detach().numpy()
        stat, pvalue = stats.levene(feature_data1, feature_data2)
        mse = np.mean((feature_data1 - feature_data2) ** 2)
        mae = np.mean(np.abs(feature_data1 - feature_data2))
        results.append({
            "Gene": gene_names[feature_idx],
            "MSE": mse,
            "MAE": mae,
            "Levene_Stat": stat,
            "P_Value": pvalue
        })
    df_results = pd.DataFrame(results)
    df_results.set_index("Gene", inplace=True)
    return df_results

def cancer_stroma_prediction_diff(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    gene_names: List[str],
    test_samples: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compute the mean difference in predictions between cancer and stroma for each gene.

    Args:
        tensor1: First tensor (samples x features).
        tensor2: Second tensor (samples x features).
        gene_names: List of gene names (length = features).
        test_samples: Optional list of sample indices to use.

    Returns:
        DataFrame with CancerStromaDiff for each gene.
    """
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape."
    assert len(gene_names) == tensor1.shape[1], "The length of gene_names must equal the number of features (columns)."
    results = []
    for feature_idx in range(tensor1.shape[1]):
        if test_samples:
            feature_data1 = tensor1[test_samples, feature_idx].detach().numpy()
            feature_data2 = tensor2[test_samples, feature_idx].detach().numpy()
        else:
            feature_data1 = tensor1[:, feature_idx].detach().numpy()
            feature_data2 = tensor2[:, feature_idx].detach().numpy()
        diff = np.mean(feature_data1 - feature_data2)
        results.append({
            "Gene": gene_names[feature_idx],
            "CancerStromaDiff": diff
        })
    df_results = pd.DataFrame(results)
    df_results.set_index("Gene", inplace=True)
    return df_results