from __future__ import annotations

import numpy as np
import torch

from autolabel.learned_fusion import invariant_token_features, pool_likelihoods
from train.model import FusionModel


def main() -> None:
    torch.manual_seed(11)
    model = FusionModel(d_model=8, heads=2).eval()
    rng = np.random.default_rng(11)
    likelihoods = [rng.dirichlet(np.ones(24)) for _ in range(3)]
    tokens = invariant_token_features(likelihoods, "key", 3, (0.4, 0.7)).astype(np.float32)
    padded_tokens = np.zeros((1, 4, 12), dtype=np.float32)
    padded_logs = np.zeros((1, 4, 24), dtype=np.float32)
    padded_tokens[0, :3] = tokens
    padded_logs[0, :3] = np.log(np.maximum(likelihoods, 1e-15))
    identities = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        logits, alpha = model.key(
            torch.tensor(padded_tokens), torch.tensor(padded_logs), identities, mask
        )
    params = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    posterior, numpy_alpha = pool_likelihoods(
        params, "key", likelihoods, [0, 1, 2], tokens, heads=2
    )
    posterior_error = float(np.max(np.abs(posterior - torch.softmax(logits[0], -1).numpy())))
    alpha_error = float(np.max(np.abs(numpy_alpha - alpha[0, :3].numpy())))
    if posterior_error >= 1e-5 or alpha_error >= 1e-6:
        raise SystemExit(f"parity failed: posterior={posterior_error}, alpha={alpha_error}")
    print(f"fusion parity ok: posterior={posterior_error:.3g}, alpha={alpha_error:.3g}")


if __name__ == "__main__":
    main()
