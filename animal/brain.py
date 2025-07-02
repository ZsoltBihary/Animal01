import torch
import torch.nn as nn
from custom_layers import SparseLinear


class BrainModel(nn.Module):
    """
    Sparse brain model that evolves internal neuron states and outputs action Q-values.

    Structure:
    - Perceptors: first `n_perceptors` neurons, overwritten by external observations.
    - Hidden neurons: evolved via a SparseLinear layer (sparse connectivity).
    - Motor neurons: final `n_motors` neurons, whose activations are further mapped to Q-values.

    Non-linearity:
    Uses sigmoid activation to clamp neuron activations for biological plausibility.

    Attributes:
        n_neurons (int): Total number of neurons in the brain.
        n_perceptors (int): Number of perceptor neurons receiving external input.
        n_motors (int): Number of motor neurons.
        n_actions (int): Number of possible actions (Q-value outputs).
        K (int): Number of incoming connections per neuron in the sparse layer.
        think_layer (SparseLinear): Sparse linear layer evolving hidden + motor neurons.
        non_linearity (nn.Module): Sigmoid activation function.
        motor_to_q (nn.Linear): Linear layer mapping motor neuron outputs to Q-values.
    """
    def __init__(self, n_neurons=1000, n_perceptors=50, n_motors=10, n_actions=5, K=20):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_perceptors = n_perceptors
        self.n_motors = n_motors
        self.n_actions = n_actions
        self.K = K
        # This layer evolves hidden + motor neurons, using all neurons as input
        self.think_layer = SparseLinear(
            in_features=n_neurons,
            out_features=n_neurons - n_perceptors,
            K=K
        )
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


class Brain:
    """
    Container managing the internal neuron state and interfacing with BrainModel
    for state updates and action value computation.

    This class maintains the internal neuron state tensor, handles injecting
    environmental observations into perceptors, and updates the state by
    invoking the BrainModel’s pure functional computations.

    Attributes:
        model (BrainModel): The brain model instance.
        state (torch.Tensor): Internal neuron activations of shape (1, n_neurons).
        n_perceptors (int): Number of perceptor neurons.
        device (str or torch.device): Device on which tensors are allocated.
    """
    def __init__(self, model: BrainModel, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.n_neurons = model.n_neurons
        self.n_perceptors = model.n_perceptors

        # Initialize state: can be zeros or small random values
        self.state = torch.zeros(1, self.n_neurons, device=self.device)

    def observe(self, observation: torch.Tensor):
        """
        Inject observation into perceptor neurons.
        Args:
            observation: torch.Tensor of shape (1, n_perceptors)
        """
        assert observation.shape == (1, self.n_perceptors), "Observation shape mismatch"
        with torch.no_grad():
            self.state[:, :self.n_perceptors] = observation.to(self.device)

    def think_and_update(self):
        """
        Perform one thinking step:
        - Pass the current full state to the model.think()
        - Update the non-perceptor part of internal state in-place with new_x
        - Return Q-values for action selection
        """
        with torch.no_grad():
            new_x, q = self.model.think(self.state)
            # Update non-perceptor neurons (slice in-place)
            self.state[:, self.n_perceptors:] = new_x
        return q


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    # Parameters
    n_neurons = 1000
    n_perceptors = 50
    n_motors = 10
    n_actions = 5
    K = 20

    # Create BrainModel and Brain
    model = BrainModel(n_neurons=n_neurons, n_perceptors=n_perceptors, n_motors=n_motors, n_actions=n_actions, K=K)
    brain = Brain(model=model, device='cpu')

    print("Initial state shape:", brain.state.shape)  # should be (1, 1000)

    # Create a dummy observation vector (batch size 1)
    obs = torch.randn(1, n_perceptors)

    # Inject observation
    brain.observe(obs)
    print("State after observation (perceptor slice):", brain.state[0, :n_perceptors])

    # Run one thinking step
    q_values = brain.think_and_update()

    print("Q-values shape:", q_values.shape)  # should be (1, n_actions)
    print("Q-values:", q_values)
    # print first 5 updated neurons
    print("State after thinking (non-perceptor slice):", brain.state[0, n_perceptors:n_perceptors+5])
