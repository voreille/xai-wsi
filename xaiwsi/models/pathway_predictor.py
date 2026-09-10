from torch import nn


class TileMLPMean(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        tile_scores = self.mlp(x)
        slide_scores = tile_scores.mean(dim=1)

        return {
            "pred": slide_scores,
            "tile_scores": tile_scores,
        }
