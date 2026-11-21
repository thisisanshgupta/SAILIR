import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sailir'))
from classifier import IBPActionClassifier

def test_box1l_classifier_uses_four_index_and_sector_widths():
    model = IBPActionClassifier(topology='box1l', prime=1009).eval()
    assert len(model.expr_enc.integral_enc.position_embeds) == 4
    assert len(model.sector_enc.position_embeddings) == 4
    assert model.action_enc.ibp_embed.num_embeddings == 4
    b, nidx, nact, nrepl = 2, 4, 4, 2
    args = (
        torch.zeros(b, 3, nidx, dtype=torch.long),
        torch.ones(b, 3, dtype=torch.long),
        torch.ones(b, 3, dtype=torch.bool),
        torch.zeros(b, 1, nidx, dtype=torch.long),
        torch.zeros(b, 1, nrepl, nidx, dtype=torch.long),
        torch.zeros(b, 1, nrepl, dtype=torch.long),
        torch.zeros(b, 1, nrepl, dtype=torch.bool),
        torch.zeros(b, 1, dtype=torch.bool),
        torch.arange(nact).repeat(b, 1),
        torch.zeros(b, nact, nidx, dtype=torch.long),
        torch.ones(b, nact, dtype=torch.bool),
        torch.tensor([[1,0,1,0],[0,1,0,1]], dtype=torch.long),
        torch.tensor([[1,0,1,0],[0,1,0,1]], dtype=torch.long),
    )
    with torch.no_grad(): logits, probabilities = model(*args)
    assert logits.shape == probabilities.shape == (b, nact)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(b), atol=1e-6)
