import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from reptile.config import Config, Observation, Action, Encoded, BrainState, PredQ, PredReward, PredObservation


class SDQNModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.model_device
        # TODO: set up layers, etc.

        # This should be at the very end of __init__
        self.to(self.device)

    def sdqn(self, encoded: Encoded, state: BrainState
             ) -> tuple[PredQ, PredReward, PredObservation, BrainState]:
        # TODO: implement
        pass

    def forward(self, encoded: Encoded, state: BrainState
                ) -> tuple[PredQ, PredReward, PredObservation]:

        pred_q, pred_r, pred_obs, _ = self.sdqn(encoded, state)
        return pred_q, pred_r, pred_obs
