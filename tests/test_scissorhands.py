import unittest

import torch

from pyramidkv.scissorhands import (
    reduce_scissorhands_scores,
    select_scissorhands_tokens,
    update_scissorhands_importance,
)


class ScissorhandsSelectionTest(unittest.TestCase):
    def test_reduce_scores_uses_recent_history_window(self):
        attn = torch.arange(24, dtype=torch.float32).view(1, 1, 4, 6)

        scores = reduce_scissorhands_scores(attn, history_window=2)

        torch.testing.assert_close(scores, attn[:, :, -2, :] + attn[:, :, -1, :])

    def test_update_importance_applies_decay(self):
        previous = torch.ones(1, 1, 4)
        current_attn = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])

        updated = update_scissorhands_importance(previous, current_attn, decay=0.5)

        torch.testing.assert_close(updated, torch.tensor([[[0.5, 1.5, 2.5, 3.5]]]))

    def test_select_tokens_keeps_sink_recent_and_top_importance(self):
        importance = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])

        indices = select_scissorhands_tokens(
            importance,
            token_budget=4,
            sink_size=1,
            recent_size=1,
        )

        torch.testing.assert_close(indices, torch.tensor([[[0, 1, 3, 5]]]))

    def test_probabilistic_selection_is_unique_and_protected(self):
        importance = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])
        generator = torch.Generator().manual_seed(11)

        indices = select_scissorhands_tokens(
            importance,
            token_budget=5,
            sink_size=1,
            recent_size=1,
            selection="prob",
            generator=generator,
        )

        self.assertEqual(tuple(indices.shape), (1, 1, 5))
        self.assertEqual(len(set(indices.flatten().tolist())), 5)
        self.assertIn(0, indices.flatten().tolist())
        self.assertIn(5, indices.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
