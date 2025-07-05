import torch
# import torch.nn as nn
from nn_model.brain_model import BrainModel


class Brain:
    """
    Container managing the internal neuron state and interfacing with BrainModel
    for state updates and action value computation.

    This class maintains the internal neuron state tensor, handles injecting
    environmental observations into perceptors, and updates the state by
    invoking the BrainModel’s pure functional computations.

    Attributes:
        model (BrainModel): The nn_model model instance.
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

    def encode_observation(self, one_hot_obs: torch.Tensor):
        # TODO: Clean up
        """
        one_hot_obs: (B, C, OW, OW)
        """
        B, C, OW, _ = one_hot_obs.shape
        flat_obs = one_hot_obs.view(B, -1)  # (B, C * OW * OW)
        self.state[:, self.perceptor_start:self.perceptor_end] = flat_obs

    def observe(self, observation: torch.Tensor):
        # TODO: obsolete, delete!
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

# TODO: We need these methods...
    # class Brain:
    #     def encode_observation(self, obs):
    #
    #     # state update
    #
    #     def decide(self):
    # # action from internal state


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
