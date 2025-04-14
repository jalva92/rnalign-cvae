"""
Interpretability example for cVAE.

This script demonstrates how to compute SHAP values for the decoder of a trained cVAE model.
"""

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cvae.data import GEPcached, setup_data_loaders
from cvae.models import CVAE
from cvae.utils import compute_decoder_shap_values

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

# Instantiate model (assume already trained and loaded)
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
# cvae.load_state_dict(torch.load("outputs/run1/results/my_model.save"))  # Uncomment to load trained model

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

# Get background and target samples
xs_bg, ys_bg = next(iter(data_loaders["test"]))
with torch.no_grad():
    z_bg, _ = cvae.get_z(xs_bg, ys_bg)
background_inputs = (z_bg, ys_bg)

xs_target, ys_target = next(iter(data_loaders["valid"]))
with torch.no_grad():
    z_target, _ = cvae.get_z(xs_target, ys_target)
target_inputs = (z_target, ys_target)

# Compute SHAP values for the decoder
shap_values = compute_decoder_shap_values(cvae, background_inputs, target_inputs)
print("SHAP values shape:", shap_values[0].shape)