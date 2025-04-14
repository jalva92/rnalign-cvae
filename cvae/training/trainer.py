"""
Training loop and core training utilities for cVAE models.
"""

import torch
import torch.nn as nn
import pyro

def run_inference_for_epoch(data_loaders, losses, cuda, eval_reg_loss=False):
    """
    Runs the inference algorithm for an epoch.
    Returns the values of all losses separately on supervised and unsupervised parts.
    """
    if cuda:
        device = "mps" if torch.backends.mps.is_available() else "cuda"
        device = "cpu" if device == "cuda" else device

    num_losses = len(losses)
    sup_batches = len(data_loaders["sup"])
    batches_per_epoch = sup_batches

    epoch_losses_sup = [0.0] * num_losses
    sup_iter = iter(data_loaders["sup"])

    for i in range(batches_per_epoch):
        xs, ys = next(sup_iter)
        if cuda:
            xs = xs.to(device)
            ys = ys.to(device)
        total_loss = losses[0].step(xs, ys)
        epoch_losses_sup[0] += total_loss
        reconloss = losses[1].evaluate_loss(xs, ys)
        epoch_losses_sup[1] += reconloss
        KLloss = losses[2].evaluate_loss(xs, ys)
        epoch_losses_sup[2] += KLloss

    return epoch_losses_sup