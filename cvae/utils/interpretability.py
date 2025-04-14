import torch
import shap

def compute_decoder_shap_values(cvae, background_inputs, target_inputs):
    """
    Compute DeepSHAP values for the decoder of a CVAE.

    Args:
        cvae: an instance of CVAE.
        background_inputs: tuple (z, ys) with background samples (torch.Tensors).
        target_inputs: tuple (z, ys) with target samples (torch.Tensors).

    Returns:
        shap_values: output from shap.DeepExplainer.shap_values.
    """
    decoder = cvae.decoder
    decoder.eval()

    def decoder_forward(inputs):
        z, ys = inputs
        if not z.requires_grad:
            z.requires_grad = True
        if not ys.requires_grad:
            ys.requires_grad = True
        return decoder([z, ys])

    explainer = shap.DeepExplainer(decoder_forward, background_inputs)
    shap_values = explainer.shap_values(target_inputs)
    return shap_values