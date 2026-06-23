import unittest

import torch

from pyramidkv.nacl import (
    reduce_nacl_proxy_scores,
    select_nacl_proxy_indices,
    select_nacl_tokens,
)


class NACLSelectionTest(unittest.TestCase):
    def test_proxy_indices_suffix_and_edges_modes(self):
        suffix = select_nacl_proxy_indices(8, proxy_size=3)
        edges = select_nacl_proxy_indices(8, proxy_size=2, mode="edges", sink_size=2)

        torch.testing.assert_close(suffix, torch.tensor([5, 6, 7]))
        torch.testing.assert_close(edges, torch.tensor([0, 1, 6, 7]))

    def test_proxy_scores_reduce_selected_query_rows(self):
        attn = torch.arange(24, dtype=torch.float32).view(1, 1, 4, 6)
        proxy = torch.tensor([1, 3])

        scores = reduce_nacl_proxy_scores(attn, proxy)

        torch.testing.assert_close(scores, attn[:, :, 1, :] + attn[:, :, 3, :])

    def test_select_tokens_keeps_proxy_and_top_scores(self):
        reduced_scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])
        proxy = torch.tensor([4, 5])

        indices = select_nacl_tokens(
            reduced_scores,
            token_budget=4,
            proxy_indices=proxy,
            random_budget=0,
        )

        torch.testing.assert_close(indices, torch.tensor([[[1, 3, 4, 5]]]))

    def test_random_budget_samples_without_duplicates_or_losing_protection(self):
        reduced_scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])
        proxy = torch.tensor([4, 5])
        generator = torch.Generator().manual_seed(7)

        indices = select_nacl_tokens(
            reduced_scores,
            token_budget=5,
            proxy_indices=proxy,
            random_budget=2,
            generator=generator,
        )

        self.assertEqual(tuple(indices.shape), (1, 1, 5))
        self.assertEqual(len(set(indices.flatten().tolist())), 5)
        self.assertTrue(set(proxy.tolist()).issubset(set(indices.flatten().tolist())))


if __name__ == "__main__":
    unittest.main()
