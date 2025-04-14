"""
Custom loss functions for cVAE training, including regularized ELBO and distance correlation.
"""

import torch
import torch.nn as nn

import pyro
from pyro.infer import Trace_ELBO

def my_cdist(x1: torch.Tensor, x2: torch.Tensor, p=2.0):
    # manual implementation of cdist which is not implemented in MPS
    # taken from https://github.com/apple/coremltools/issues/2198
    assert p == 2.0

    x1_norm = x1.pow(2).sum(-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)

    x1_ = torch.cat([x1.mul(-2), x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)

    result = x1_.matmul(x2_.transpose(0, 1)) 
    result = result.clamp_min_(1e-10).sqrt_()
    return result

def distance_correlation(X, Y):
    """
    Compute the distance correlation between two matrices X and Y.
    Args:
        X (torch.Tensor): A 2D tensor of shape (n, p), where n is the number of samples and p is the number of features.
        Y (torch.Tensor): A 2D tensor of shape (n, q), where n is the number of samples and q is the number of features.
    Returns:
        float: The distance correlation between X and Y.
    """
    n = X.size(0)
    X_dist = my_cdist(X, X, p=2)
    Y_dist = my_cdist(Y, Y, p=2)
    X_mean = X_dist.mean(dim=0, keepdim=True)
    Y_mean = Y_dist.mean(dim=0, keepdim=True)
    X_double_centered = X_dist - X_mean - X_mean.T + X_dist.mean()
    Y_double_centered = Y_dist - Y_mean - Y_mean.T + Y_dist.mean()
    dcov_XY = torch.sqrt((X_double_centered * Y_double_centered).mean())
    dcov_X = torch.sqrt((X_double_centered * X_double_centered).mean())
    dcov_Y = torch.sqrt((Y_double_centered * Y_double_centered).mean())
    dcor = dcov_XY / torch.sqrt(dcov_X * dcov_Y)
    return dcor

class ysregularizedELBO(Trace_ELBO):
    def __init__(self, grad_reg_multiplier=1, cor_reg_multiplier=1, grad_reg_epsilon=0.1, *args, **kwargs):
        super().__init__(retain_graph=True, *args, **kwargs) # ensures graph is retained for second and third backprop
        self.grad_reg_multiplier = grad_reg_multiplier
        self.cor_reg_multiplier = cor_reg_multiplier
        self.grad_reg_epsilon = grad_reg_epsilon

    def loss_and_grads(self, model, guide, *args, **kwargs):
        # Ensure ys requires gradients
        ys = args[1]
        ys.requires_grad = True

        nsamples = ys.shape[0]
        nfeatures = args[0].shape[1]

        # 1. ELBO
        # Compute ELBO without applying gradients
        base_loss = super().loss_and_grads(model, guide, *args, **kwargs)

        # 2. Gradient norm regularization
        # Compute norm of gradients of ys
        reg_loss = -1 * self.grad_reg_multiplier * torch.norm(ys.grad) * nsamples * nfeatures
        reg_loss.requires_grad = True
        reg_loss.backward(retain_graph=True)

        # 2. Calculate correlation penalty as before
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        zs = guide_trace.nodes["z"]["value"]
        nzdims = zs.shape[1]
        correlation_penalty = distance_correlation(zs, ys)
        scaled_correlation_penalty = correlation_penalty * self.cor_reg_multiplier / nzdims #zdim normalisation
        scaled_correlation_penalty.backward() # retain_graph=True

        # Combine all loss components: base ELBO + gradient regularization + correlations
        total_loss = base_loss + reg_loss + scaled_correlation_penalty

        # Backward pass applied *once* on total_loss
        ys.grad = None
        reg_loss.grad = None
        scaled_correlation_penalty.grad = None

        # print out for capture separate from the returned total_loss value
        print(str(base_loss)+','+str((reg_loss).cpu().detach().numpy())+','+str((scaled_correlation_penalty).cpu().detach().numpy()))

        return total_loss