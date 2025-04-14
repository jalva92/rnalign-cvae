"""
Preprocessing utilities for cVAE gene expression models.

Includes functions for scaling expression data, processing purity, and encoding class labels.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def scale_expression_data(expression_csv: str, device: str = "cpu"):
    """
    Load and z-score scale expression data from a CSV file.
    Returns:
        test_scaled (pd.DataFrame): Scaled expression data.
        data (torch.Tensor): Scaled data as torch tensor.
        genedf (pd.DataFrame): Alias for test_scaled.
        num_genes (int): Number of genes/features.
    """
    std_scaler = StandardScaler()
    test_full = pd.read_csv(expression_csv, index_col=0)
    test_scaled = pd.DataFrame(std_scaler.fit_transform(test_full), columns=test_full.columns)
    test_scaled.index = test_full.index
    data = torch.tensor(test_scaled.values, dtype=torch.float32, device=device)
    genedf = test_scaled
    num_genes = data.size()[1]
    return test_scaled, data, genedf, num_genes, std_scaler

def process_purity(purity_csv: str, data, ysmultiplier=1.0, scaleys=False, device="cpu"):
    """
    Load and process purity values from a CSV file.
    Returns:
        purity_levels (torch.Tensor): Processed purity values.
        classdf (torch.Tensor): Appended to classdf.
    """
    purity_std_scaler = StandardScaler()
    purities_input = pd.read_csv(purity_csv, index_col=0)
    if scaleys:
        purities_scaled = pd.DataFrame(purity_std_scaler.fit_transform(purities_input), columns=purities_input.columns)
        purities_scaled.index = purities_input.index
        purities_test = purities_scaled * ysmultiplier
    else:
        purities_test = purities_input * ysmultiplier

    purity_levels = torch.tensor(purities_test['purity'].values, dtype=torch.float32, device=device)
    mean_value = torch.nanmean(purity_levels)
    nan_mask = torch.isnan(purity_levels)
    purity_levels[nan_mask] = mean_value
    purity_levels = purity_levels.unsqueeze(1)
    classdf = torch.cat([torch.zeros((data.size(0), 0), dtype=data.dtype, device=device), purity_levels], dim=1)
    return purity_levels, classdf

def process_classes(class_csv: str, data, device="cpu"):
    """
    Load and one-hot encode class labels from a CSV file.
    Returns:
        cancer_types (torch.Tensor): One-hot encoded class labels.
        classdf (torch.Tensor): Appended to classdf.
        stratification_classes (np.ndarray): For stratified splitting.
        ohe (OneHotEncoder): Fitted encoder.
        cancer_classes (pd.DataFrame): Original class dataframe.
    """
    ohe = OneHotEncoder(sparse_output=False)
    cancer_classes = pd.read_csv(class_csv, index_col=0)
    encoded_disease = ohe.fit_transform(cancer_classes)
    cancer_types = torch.tensor(encoded_disease, dtype=torch.float32, device=device)
    classdf = torch.cat([torch.zeros((data.size(0), 0), dtype=data.dtype, device=device), cancer_types], dim=1)
    stratification_classes = encoded_disease[:, [0, 1]] if encoded_disease.shape[1] >= 2 else None
    return cancer_types, classdf, stratification_classes, ohe, cancer_classes