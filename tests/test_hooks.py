import unittest
from unittest.mock import patch

import torch

from src.inspect_model import activation_norms, group_table, parameter_rows


class Inputs(dict):
    def to(self, device):
        return self


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.device = torch.device("cpu")
        self.fail = False

    def get_decoder(self):
        return self

    def forward(self, input_ids):
        x = input_ids.float().unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
            if self.fail:
                raise RuntimeError("forward failed")
        return x


class HooksTest(unittest.TestCase):
    def test_repeat_and_exception_restore_state(self):
        model = Model().train()
        model.layers[1].eval()
        existing = model.layers[0].register_forward_hook(lambda *args: None)
        flags = [m.training for m in model.modules()]
        hooks = [len(m._forward_hooks) for m in model.modules()]
        tokenizer = lambda *a, **kw: Inputs(input_ids=torch.tensor([[1, 2, 3]]))
        params = {"hooks": {"prompt": "test"}}
        try:
            with patch("src.inspect_model.build_prompt", return_value="test"):
                first = activation_norms(tokenizer, model, params)
                self.assertEqual(first, activation_norms(tokenizer, model, params))
                self.assertEqual(flags, [m.training for m in model.modules()])
                self.assertEqual(hooks, [len(m._forward_hooks) for m in model.modules()])
                model.fail = True
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    activation_norms(tokenizer, model, params)
            self.assertEqual(flags, [m.training for m in model.modules()])
            self.assertEqual(hooks, [len(m._forward_hooks) for m in model.modules()])
        finally:
            existing.remove()

    def test_tied_and_untied_head(self):
        for tied in (False, True):
            model = torch.nn.Module()
            model.embed_tokens = torch.nn.Embedding(7, 3)
            model.lm_head = torch.nn.Linear(3, 7, bias=False)
            if tied:
                model.lm_head.weight = model.embed_tokens.weight
            table = group_table(parameter_rows(model))
            self.assertEqual(sum(g["params"] for g in table),
                             sum(p.numel() for p in model.parameters()))
            self.assertEqual(table[-1]["params"], 0 if tied else 21)


if __name__ == "__main__":
    unittest.main()
