"""
Custom dataset example for cVAE.

This script demonstrates how to use a custom gene expression dataset with the cVAE API.
"""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from cvae.data import GEPcached, setup_data_loaders
from cvae.models import CVAE

# Load your custom expression data
expressionfile = "data/custom_expression.csv"
expr_df = pd.read_csv(expressionfile, index_col=0)
scaler = StandardScaler()
expr_scaled = pd.DataFrame(scaler.fit_transform(expr_df), columns=expr_df.columns)
expr_scaled.index = expr_df.index
data = torch.tensor(expr_scaled.values, dtype=torch.float32)
num_genes = data.size()[1]

# Load and encode custom class labels (if available)
classfile = "data/custom_class.csv"
if classfile != ".":
    ohe = OneHotEncoder(sparse_output=False)
    class_df = pd.read_csv(classfile, index_col=0)
    encoded_classes = ohe.fit_transform(class_df)
    classdf = torch.tensor(encoded_classes, dtype=torch.float32)
    num_classes = classdf.size(1)
else:
    classdf = torch.zeros((data.size(0), 0), dtype=data.dtype)
    num_classes = 0

# Instantiate model
cvae = CVAE(
    z_dim=8,
    input_size=num_genes,
    output_size=num_classes,
    hidden_layers=[32, 16],
    use_cuda=False,
    annealingfactor=1.0,
    decoder_sigma=1.0,
    decoder_batchnorm=True,
    latentstd=0.01,
    dropoutp=0.1
)

# Data loaders
data_loaders = setup_data_loaders(
    dataset=GEPcached, use_cuda=False, batch_size=16,
    sup_num=80, val_num=20,
    input_GEP=expr_scaled,
    input_classes=classdf,
    num_classes=num_classes,
    train_data_size=len(expr_scaled),
    stratifyclasses=None
)

print("Custom dataset loaded and model initialized.")