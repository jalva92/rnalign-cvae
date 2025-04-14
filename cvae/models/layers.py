"""
Custom MLP and layer modules for cVAE models.

Includes MLP, activation modules, and utility classes for flexible neural network construction.
"""

from inspect import isclass

import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape

class Exp(nn.Module):
    """A custom module for exponentiation of tensors."""
    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)

class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        softplus_output = self.softplus(x)
        return softplus_output

class SoftplusWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(SoftplusWithDropout, self).__init__()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        softplus_output = self.softplus(x)
        dropout_output = self.dropout(softplus_output)
        return dropout_output

class ScaledPuritySoftplusWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ScaledPuritySoftplusWithDropout, self).__init__()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def forward(self, x):
        # Apply Softplus and Dropout to all features
        x = self.softplus(x)
        x = self.dropout(x)

        # Extract the last feature
        last_feature = x[:, -1]

        # Update min and max values
        with torch.no_grad():
            self.min_val = torch.min(self.min_val, last_feature.min())
            self.max_val = torch.max(self.max_val, last_feature.max())

        # Perform min-max scaling on the last feature
        scaled_last_feature = (last_feature - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # Replace the last feature with the scaled version
        x[:, -1] = scaled_last_feature

        return x

    def reset_parameters(self):
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))

class ConcatModule(nn.Module):
    """A custom module for concatenation of tensors."""
    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        if len(input_args) == 1:
            input_args = input_args[0]
        if torch.is_tensor(input_args):
            return input_args
        else:
            input_args = [t for t in input_args if t.size(1) > 0]
            if not input_args:
                return torch.tensor([])
            if len(input_args) == 1:
                return input_args[0]
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)

class ListOutModule(nn.ModuleList):
    """A custom module for outputting a list of tensors from a list of nn modules."""
    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return [mm.forward(*args, **kwargs) for mm in self]

def call_nn_op(op):
    """
    Helper function to add appropriate parameters when calling
    an nn module representing an operation like Softmax.
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()

class MLP(nn.Module):
    """
    Flexible multi-layer perceptron supporting custom activations, batchnorm, and output heads.
    """
    def __init__(
        self,
        mlp_sizes,
        activation=nn.ReLU,
        output_activation=None,
        post_layer_fct=lambda layer_ix, total_layers, layer: None,
        post_act_fct=lambda layer_ix, total_layers, layer: None,
        allow_broadcast=False,
        use_cuda=False,
        bn=False,
    ):
        super().__init__()

        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        if bn:
            bn_layer = nn.BatchNorm1d(mlp_sizes[0])
            self.bn = bn_layer
        else:
            self.bn = None

        input_size, hidden_sizes, output_size = (
            mlp_sizes[0],
            mlp_sizes[1:-1],
            mlp_sizes[-1],
        )

        assert isinstance(input_size, (int, list, tuple)), "input_size must be int, list, tuple"

        last_layer_size = input_size if type(input_size) == int else sum(input_size)
        all_modules = [ConcatModule(allow_broadcast)]

        if bn:
            all_modules.append(bn_layer)

        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, "Hidden layer sizes must be ints"
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)
            all_modules.append(cur_linear_layer)
            post_linear = post_layer_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )
            if post_linear is not None:
                all_modules.append(post_linear)
            all_modules.append(activation())
            post_activation = post_act_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )
            if post_activation is not None:
                all_modules.append(post_activation)
            last_layer_size = layer_size

        assert isinstance(output_size, (int, list, tuple)), "output_size must be int, list, tuple"

        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(
                    call_nn_op(output_activation)
                    if isclass(output_activation)
                    else output_activation
                )
        else:
            out_layers = []
            for out_ix, out_size in enumerate(output_size):
                split_layer = []
                split_layer.append(nn.Linear(last_layer_size, out_size))
                act_out_fct = (
                    output_activation
                    if not isinstance(output_activation, (list, tuple))
                    else output_activation[out_ix]
                )
                if act_out_fct:
                    split_layer.append(
                        call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct
                    )
                out_layers.append(nn.Sequential(*split_layer))
            all_modules.append(ListOutModule(out_layers))

        self.sequential_mlp = nn.Sequential(*all_modules)

    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)