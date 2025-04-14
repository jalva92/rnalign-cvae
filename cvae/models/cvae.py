"""
Conditional Variational Autoencoder (cVAE) implementation using Pyro.

This module defines the CVAE model for gene expression data, supporting
continuous and categorical conditioning, and modular MLP-based encoder/decoder.
"""

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

import torch
import torch.nn as nn
from .layers import MLP, Exp, Softplus, SoftplusWithDropout, ScaledPuritySoftplusWithDropout

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for gene expression data.

    Args:
        output_size (int): Number of output features (e.g., purity + classes).
        input_size (int): Number of input features (genes).
        z_dim (int): Latent dimension size.
        hidden_layers (list): List of hidden layer sizes.
        config_enum (str, optional): Parallel config for broadcasting.
        use_cuda (bool): Whether to use CUDA/MPS.
        annealingfactor (float): KL annealing factor.
        decoder_sigma (float): Initial sigma for decoder output.
        latentstd (float): Latent stddev scaling.
        decoder_batchnorm (bool): Use batchnorm in decoder.
        output_sigmoid (bool): Use sigmoid at decoder output.
        encoderpuritynorm (bool): Use scaled purity in encoder.
        dropoutp (float): Dropout probability.
    """
    def __init__(
        self,
        output_size=1,
        input_size=3000,
        z_dim=64,
        hidden_layers=[1500, 750],
        config_enum=None,
        use_cuda=False,
        annealingfactor=1,
        decoder_sigma=1,
        latentstd=1,
        decoder_batchnorm=False,
        output_sigmoid=False,
        encoderpuritynorm=False,
        dropoutp=0.5,
    ):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == "parallel"
        self.use_cuda = use_cuda
        self.decoder_batchnorm = decoder_batchnorm
        self.annealingfactor = annealingfactor
        self.sharedsigma = decoder_sigma
        self.outputsigmoid = output_sigmoid
        self.outputactivationdecoder = None
        self.encoderpuritynorm = encoderpuritynorm
        self.inputactivationencoder = SoftplusWithDropout
        self.latentstd = latentstd
        self.dropoutp = dropoutp

        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        if self.outputsigmoid:
            self.outputactivationdecoder = nn.Sigmoid
        if self.encoderpuritynorm:
            self.inputactivationencoder = ScaledPuritySoftplusWithDropout

        self.encoder_z = MLP(
            [self.input_size + self.output_size] + hidden_sizes + [[z_dim, z_dim]],
            activation=nn.ReLU,
            output_activation=[None, Softplus],
            post_act_fct=lambda layer_ix, total_layers, layer: nn.Dropout(p=self.dropoutp) if layer_ix < total_layers else None,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.decoder = MLP(
            [z_dim + self.output_size] + hidden_sizes + [self.input_size],
            activation=nn.ReLU,
            output_activation=self.outputactivationdecoder,
            post_act_fct=lambda layer_ix, total_layers, layer: nn.Dropout(p=self.dropoutp) if layer_ix < total_layers else None,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
            bn=self.decoder_batchnorm,
        )

        if self.use_cuda:
            device = "mps" if torch.backends.mps.is_available() else "cuda"
            self.to(device)

    def model(self, xs, ys=None):
        pyro.module("cvae", self)
        batch_size = xs.shape[0]
        options = dict(dtype=xs.dtype, device=xs.device)

        with pyro.plate("data"):
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            reparamscale_tensor = torch.tensor(self.latentstd, dtype=torch.float32, device=prior_loc.device)
            z_scale = prior_scale * reparamscale_tensor

            with pyro.poutine.scale(scale=self.annealingfactor):
                zs = pyro.sample("z", dist.Normal(prior_loc, z_scale).to_event(1))

            loc = self.decoder([zs, ys])
            scale = pyro.param("scale", torch.ones_like(loc, **options) * self.sharedsigma, constraint=constraints.positive)
            pyro.sample("obs", dist.Normal(loc, scale, validate_args=False).to_event(1), obs=xs)
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            loc, raw_scale = self.encoder_z([xs, ys])
            scale = torch.nn.functional.softplus(raw_scale) + 1e-6
            with pyro.poutine.scale(scale=self.annealingfactor):
                pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def get_z(self, xs, ys):
        encoder_input = [xs, ys]
        zmean, zvar = self.encoder_z(encoder_input)
        return zmean, zvar

    def recon_fn(self, xs, ys):
        zmean, _ = self.get_z(xs, ys)
        xhat = self.decoder([zmean, ys])
        return xhat