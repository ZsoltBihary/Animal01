import torch
import torch.nn.functional as F
# import torch.nn as nn
from nn_model.brain_model import BrainModel
from utils.helper import N_CELL_TYPES, N_ACTIONS, BRAIN_COST


class Brain:
    """
    Container managing the internal neuron activation and interfacing with BrainModel
    for activation updates and action value computation.

    This class maintains the internal neuron activation tensor, handles injecting
    environmental observations into perceptors, and updates the activation by
    invoking the BrainModel’s pure functional computations.

    Attributes:
        model (BrainModel): The nn_model model instance.
        activation (torch.Tensor): Internal neuron activations of shape (B, n_neurons).
        n_perceptors (int): Number of perceptor neurons.
        device (str or torch.device): Device on which tensors are allocated.
    """
    def __init__(self, B: int, model: BrainModel, device='cpu'):
        self.B = B
        self.model = model.to(device)
        self.device = device
        self.n_neurons = model.n_neurons
        self.n_perceptors = model.n_perceptors
        self.n_motors = model.n_motors
        self.n_connections = model.n_connections
        self.n_actions = model.n_actions

        # Initialize activation: can be zeros or small random values
        self.activation = torch.zeros(self.B, self.n_neurons, device=self.device)

    def encode_observation(self, obs: torch.Tensor):
        # TODO: Clean up
        """
        obs: (B, K, K)
        """
        # B, K, _ = obs.shape
        # One-hot encode (skip EMPTY = 0)
        onehot = F.one_hot(obs, num_classes=N_CELL_TYPES)[..., 1:]  # (B, K, K, C)
        perception = onehot.flatten(1).float()                      # (B, K*K*C)
        self.activation[:, :self.n_perceptors] = perception

    def think(self):
        """
        Perform one thinking step:
        - Pass the current full activation to the model.think()
        - Update the non-perceptor part of internal activation in-place with new_x
        - Return Q-values for action selection
        """
        with torch.no_grad():
            new_activation, q = self.model.think(self.activation)
            # Update non-perceptor neurons (slice in-place)
            self.activation[:, self.n_perceptors:] = new_activation
            energy_cost = BRAIN_COST * new_activation.sum(dim=1)
        return q, energy_cost


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    # Parameters
    B = 3
    R = 2
    K = R*2+1
    n_neurons = 1000
    n_perceptors = K*K*(N_CELL_TYPES-1)
    n_motors = 10
    n_actions = N_ACTIONS
    n_connections = 20

    # Create BrainModel and Brain
    model = BrainModel(n_neurons=n_neurons, n_perceptors=n_perceptors, n_motors=n_motors, n_actions=n_actions,
                       n_connections=n_connections)
    brain = Brain(B=B, model=model, device='cpu')

    print("Initial activation shape:", brain.activation.shape)  # should be (1, 1000)

    # Create a dummy observation vector (batch size 1)

    obs = torch.randint(low=0, high=N_CELL_TYPES, size=(B, K, K))

    # Inject observation
    brain.encode_observation(obs)
    print("State after observation (perceptor slice):", brain.activation[0, :n_perceptors])

    # Run one thinking step
    q_values, energy_cost = brain.think()

    print("Q-values shape:", q_values.shape)  # should be (B, n_actions)
    print("Q-values:", q_values)
    # print first 5 updated neurons
    print("State after thinking (non-perceptor slice):", brain.activation[0, n_perceptors:n_perceptors + 5])
