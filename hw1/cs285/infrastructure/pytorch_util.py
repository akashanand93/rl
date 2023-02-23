from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation=nn.ReLU(),
                 output_activation=nn.ReLU()):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        layers = [self.input_layer, self.relu]
        for i in range(num_hidden_layers):
            layers.append(self.hidden_layers[i])
            layers.append(activation)
        layers.append(self.output_layer)
        layers.append(output_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    return MLP(input_dim=input_size,
               hidden_dim=size,
               output_dim=output_size,
               num_hidden_layers=n_layers,
               activation=activation,
               output_activation=output_activation)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
