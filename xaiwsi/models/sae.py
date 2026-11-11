"""TopK sparse autoencoder on tile embeddings (Gao et al. 2024, "Scaling and evaluating sparse
autoencoders").

``z = TopK(ReLU(W_enc (x - b_pre) + b_enc))`` keeps the k largest latents per tile, and
``x_hat = z D + b_pre`` with unit-norm decoder rows ``D`` (K x D_in). Inputs are standardised
with a fixed per-dimension mean and one global scale (``set_input_stats``), an affine map,
so the slide mean of the inputs is still recovered linearly from the slide mean of the codes:
``mean_i x_i ≈ (mean_i z_i) D * scale + mean + b_pre * scale``.

Dead latents (no activation for ``dead_after`` training tiles) get the auxiliary loss of Gao
et al.: the residual is reconstructed from the ``aux_k`` largest dead pre-activations, with
weight ``aux_coef``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_latents: int,
        k: int,
        aux_k: int = 512,
        aux_coef: float = 1 / 32,
        dead_after: int = 10_000_000,
    ) -> None:
        super().__init__()
        self.input_dim, self.n_latents, self.k = input_dim, n_latents, k
        self.aux_k, self.aux_coef, self.dead_after = aux_k, aux_coef, dead_after
        self.encoder = nn.Linear(input_dim, n_latents)
        self.decoder = nn.Parameter(torch.empty(n_latents, input_dim))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.decoder)
        with torch.no_grad():
            self.decoder.div_(self.decoder.norm(dim=1, keepdim=True))
            # Encoder initialised as the decoder transpose (Gao et al.).
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("tiles_since_active", torch.zeros(n_latents, dtype=torch.long))

    def set_input_stats(self, mean: torch.Tensor, scale: float) -> None:
        self.input_mean.copy_(mean)
        self.input_scale.fill_(scale)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.input_mean) / self.input_scale

    def pre_activations(self, x_norm: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x_norm - self.pre_bias))

    def topk(self, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, indices = pre.topk(self.k, dim=-1)
        return values, indices

    def sparse_matmul(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Sum of the selected decoder rows, weighted by ``values`` (B x k each).

        Scatters into dense B x K codes and uses a plain matmul: the gather-einsum form
        dispatches its backward to a Triton kernel in recent torch, which needs the Python
        headers at runtime (missing on the GPU hosts).
        """
        codes = torch.zeros(len(values), self.n_latents, device=values.device, dtype=values.dtype)
        return codes.scatter(1, indices, values) @ self.decoder

    def decode_sparse(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.sparse_matmul(values, indices) + self.pre_bias

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw tile embeddings (B x D) -> top-k (values, indices), both B x k."""
        return self.topk(self.pre_activations(self.normalize(x)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Loss terms on a batch of raw tile embeddings; updates the dead-latent counters."""
        x_norm = self.normalize(x)
        pre = self.pre_activations(x_norm)
        values, indices = self.topk(pre)
        x_hat = self.decode_sparse(values, indices)
        residual = x_norm - x_hat
        mse = residual.pow(2).sum(-1).mean()
        # Fraction of variance unexplained, against the batch mean.
        fvu = mse / (x_norm - x_norm.mean(0)).pow(2).sum(-1).mean()

        if self.training:
            active = torch.zeros(self.n_latents, dtype=torch.bool, device=x.device)
            active[indices[values > 0]] = True
            self.tiles_since_active += len(x)
            self.tiles_since_active[active] = 0
        dead = self.tiles_since_active >= self.dead_after
        aux = torch.zeros((), device=x.device)
        n_dead = int(dead.sum())
        if self.training and n_dead > 0 and self.aux_coef > 0:
            aux_pre = pre.masked_fill(~dead, 0.0)
            aux_values, aux_indices = aux_pre.topk(min(self.aux_k, n_dead), dim=-1)
            aux_hat = self.sparse_matmul(aux_values, aux_indices)
            aux = (residual.detach() - aux_hat).pow(2).sum(-1).mean() / residual.detach().pow(2).sum(-1).mean()
        return {
            "loss": mse + self.aux_coef * aux,
            "mse": mse.detach(),
            "fvu": fvu.detach(),
            "aux": aux.detach(),
            "dead_fraction": torch.tensor(n_dead / self.n_latents),
        }

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Back to unit-norm decoder rows, after each optimizer step."""
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True))

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove the gradient component parallel to the decoder rows, before each step."""
        if self.decoder.grad is None:
            return
        parallel = (self.decoder.grad * self.decoder).sum(dim=1, keepdim=True)
        self.decoder.grad.sub_(parallel * self.decoder)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "n_latents": self.n_latents,
            "k": self.k,
            "aux_k": self.aux_k,
            "aux_coef": self.aux_coef,
            "dead_after": self.dead_after,
        }


def load_sae(path, map_location="cpu") -> TopKSAE:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model = TopKSAE(**state["config"])
    model.load_state_dict(state["state_dict"])
    return model.eval()
