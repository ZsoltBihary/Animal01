# import torch
from torch import Tensor
import torch.nn as nn
from nn_model.custom_layers import SparseLinear


class BrainModel(nn.Module):
    """
    Sparse nn_model model that evolves internal neuron states and outputs action Q-values.

    Structure:
    - Perceptors: first `n_perceptors` neurons, overwritten by external observations.
    - Hidden neurons: evolved via a SparseLinear layer (sparse connectivity).
    - Motor neurons: final `n_motors` neurons, whose activations are further mapped to Q-values.

    Non-linearity:
    Uses sigmoid activation to clamp neuron activations for biological plausibility.

    Attributes:
        n_neurons (int): Total number of neurons in the nn_model.
        n_perceptors (int): Number of perceptor neurons receiving external input.
        n_motors (int): Number of motor neurons.
        n_actions (int): Number of possible actions (Q-value outputs).
        n_connections (int): Number of incoming connections per neuron in the sparse layer.
        think_layer (SparseLinear): Sparse linear layer evolving hidden + motor neurons.
        non_linearity (nn.Module): Sigmoid activation function.
        motor_to_q (nn.Linear): Linear layer mapping motor neuron outputs to Q-values.
    """
    def __init__(self, n_neurons: int, n_perceptors: int, n_motors: int, n_connections: int, n_actions: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_perceptors = n_perceptors
        self.n_motors = n_motors
        self.n_connections = n_connections
        self.n_actions = n_actions

        # This layer evolves hidden + motor neurons, using all neurons as input
        self.think_layer = SparseLinear(in_features=n_neurons, out_features=n_neurons - n_perceptors,
                                        connections=n_connections)
        # Non-linearity
        self.non_linearity = nn.Sigmoid()
        # This layer maps motor neurons to Q-values (unbounded)
        self.motor_to_q = nn.Linear(n_motors, n_actions)

    def think(self, x):
        """
        Calculate new hidden and motor neuron activations, and action values.
        Input: x(batch, n_neurons)
        Output: new_x(batch, n_neurons-n_perceptors), Q-values(batch, n_actions)
        """
        new_x = self.think_layer(x)
        new_x = self.non_linearity(new_x)
        q = self.motor_to_q(new_x[:, -self.n_motors:])
        return new_x, q

    def forward(self, x):
        """
        Forward pass for training.
        Input: x(batch, n_neurons)
        Output: Q-values(batch, n_actions)
        """
        _, q = self.think(x)
        return q

    def rewire(self, index_tensor: Tensor):
        self.think_layer.set_index_tensor(index_tensor)
