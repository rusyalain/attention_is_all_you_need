"""Small CPU checks of causality, gradients, learning and checkpoint loading."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from ruslit_lm import Config, Decoder


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(20261109)
        self.config = Config(vocab_size=97, context_length=16, layers=2,
                             width=32, heads=4, dropout=0)

    def test_causality_and_manual_sdpa_gradients(self):
        fast = Decoder(self.config).eval()
        manual = Decoder(Config(**(asdict(self.config) | {'attention': 'manual'}))).eval()
        manual.load_state_dict(fast.state_dict())
        ids = torch.randint(0, 97, (2, 16))
        targets = torch.randint(0, 97, (2, 16))
        changed = ids.clone()
        changed[:, 8:] = (changed[:, 8:] + 11) % 97
        logits, loss = fast(ids, targets)
        reference, reference_loss = manual(ids, targets)
        self.assertLess((logits[:, :8] - fast(changed)[0][:, :8]).abs().max().item(), 1e-7)
        self.assertLess((logits - reference).abs().max().item(), 1e-5)
        loss.backward()
        reference_loss.backward()
        error = max((p.grad - q.grad).abs().max().item()
                    for p, q in zip(fast.parameters(), manual.parameters()))
        self.assertLess(error, 1e-5)
        for weights in fast.attention_maps(ids):
            self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)

    def test_learning_on_small_sequence(self):
        model = Decoder(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        sequence = torch.arange(17).repeat(2, 1) % 7
        x, y = sequence[:, :-1], sequence[:, 1:]
        initial = model(x, y)[1].item()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), initial * .7)

    def test_checkpoint_roundtrip(self):
        model = Decoder(self.config).eval()
        ids = torch.randint(0, 97, (2, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'config': asdict(self.config), 'model': model.state_dict()}, path)
            state = torch.load(path, map_location='cpu', weights_only=True)
            loaded = Decoder(Config(**state['config'])).eval()
            loaded.load_state_dict(state['model'])
            self.assertLess((model(ids)[0] - loaded(ids)[0]).abs().max().item(), 1e-7)

    def test_main_parameter_count(self):
        self.assertEqual(Decoder(Config()).parameter_count, 29_676_544)


if __name__ == '__main__':
    unittest.main()
