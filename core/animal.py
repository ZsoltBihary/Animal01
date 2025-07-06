# animal.py
from core.policy import BasePolicy


class Animal:
    # TODO: This class is very thin, perhaps unnecessary???
    def __init__(self, policy: BasePolicy):
        """
        Args:
            policy: An instance of BasePolicy (e.g., RandomPolicy, BrainPolicy)
        """
        self.policy = policy

    # def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
    #     """
    #     Choose action based on observation.
    #     Args:
    #         obs: Tensor of shape (B, n_connections, n_connections)
    #     Returns:
    #         actions: Tensor of shape (B,)
    #         brain_cost: Tensor of shape (B,)  (0.0 if no brain used)
    #     """
    #     return self.policy(obs)
