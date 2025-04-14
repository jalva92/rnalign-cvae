"""
Basic training example for cVAE.

This script demonstrates how to train a cVAE model using the Python API.
"""

from cvae.data import GEPcached, setup_data_loaders
from cvae.models import CVAE
from cvae.training import ysregularizedELBO
from cvae.utils.metrics import get_mae, get_mse
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
expressionfile = "data/example_expression.csv"
test_full = pd.read_csv(expressionfile, index_col=0)
std_scaler = StandardScaler()
test_scaled = pd.DataFrame(std_scaler.fit_transform(test_full), columns=test_full.columns)
test_scaled.index = test_full.index
data = torch.tensor(test_scaled.values, dtype=torch.float32)
genedf = test_scaled
num_genes = data.size()[1]

# Dummy class info
classdf = torch.zeros((data.size(0), 0), dtype=data.dtype)
num_classes = 0

# Model
cvae = CVAE(
    z_dim=16,
    input_size=num_genes,
    output_size=num_classes,
    hidden_layers=[64, 32],
    use_cuda=False,
    annealingfactor=1.0,
    decoder_sigma=1.0,
    decoder_batchnorm=False,
    latentstd=0.01,
    dropoutp=0.0
)

# Data loaders
data_loaders = setup_data_loaders(
    dataset=GEPcached, use_cuda=False, batch_size=32,
    sup_num=100, val_num=20,
    input_GEP=genedf,
    input_classes=classdf,
    num_classes=num_classes,
    train_data_size=len(test_scaled),
    stratifyclasses=None
)

# Training loop (minimal)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro

optimizer = Adam({"lr": 1e-3})
svi = SVI(cvae.model, cvae.guide, optimizer, loss=Trace_ELBO())

for epoch in range(10):
    loss = svi.step(data_loaders["train"].dataset.data, data_loaders["train"].dataset.targets)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# Evaluate
mae = get_mae(data_loaders["test"], cvae.recon_fn, 32)
mse = get_mse(data_loaders["test"], cvae.recon_fn, 32)
print(f"Test MAE: {mae}, Test MSE: {mse}")