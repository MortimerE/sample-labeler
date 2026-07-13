import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autolabel.learned_fusion import invariant_token_features, pool_likelihoods
from train.model import FusionModel


def test_torch_numpy_attention_forward_parity():
    torch.manual_seed(11)
    model = FusionModel(d_model=8, heads=2).eval()
    rng = np.random.default_rng(11)
    likelihoods = [rng.dirichlet(np.ones(24)) for _ in range(3)]
    tokens = invariant_token_features(likelihoods, "key", 3, (0.4, 0.7)).astype(np.float32)
    padded_tokens = np.zeros((1, 4, 12), dtype=np.float32)
    padded_logs = np.zeros((1, 4, 24), dtype=np.float32)
    padded_tokens[0, :3] = tokens
    padded_logs[0, :3] = np.log(np.maximum(likelihoods, 1e-15))
    with torch.no_grad():
        logits, alpha = model.key(
            torch.tensor(padded_tokens), torch.tensor(padded_logs),
            torch.tensor([[0, 1, 2, 3]]), torch.tensor([[True, True, True, False]]),
        )
    params = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    posterior, numpy_alpha = pool_likelihoods(params, "key", likelihoods, [0, 1, 2], tokens, heads=2)
    assert posterior == pytest.approx(torch.softmax(logits[0], -1).numpy(), abs=1e-5)
    assert numpy_alpha == pytest.approx(alpha[0, :3].numpy(), abs=1e-6)
