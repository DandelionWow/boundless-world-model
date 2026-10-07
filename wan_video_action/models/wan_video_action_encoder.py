import torch
import torch.nn as nn


class WanVideoActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        dim: int = 1536,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.action_mlp1 = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_mlp2 = nn.Sequential(
            nn.Linear(action_dim * 4, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, action):
        action_context_emb = self.action_mlp1(action)
        grouped_action = torch.cat([action[:, 0:1].repeat(1, 3, 1), action], dim=1)
        grouped_action = grouped_action.reshape(action.shape[0], (action.shape[1] + 3) // 4, action.shape[2] * 4)
        action_mod_emb = self.action_mlp2(grouped_action)
        return action_context_emb, action_mod_emb

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        incompatible_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing_keys = [
            key for key in incompatible_keys.missing_keys
            if not key.startswith("action_mlp1.") and not key.startswith("action_mlp2.")
        ]
        if strict and (missing_keys or incompatible_keys.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for WanVideoActionEncoder: "
                f"missing_keys={missing_keys}, unexpected_keys={list(incompatible_keys.unexpected_keys)}"
            )
        return incompatible_keys
