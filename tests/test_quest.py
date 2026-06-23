import unittest

import torch

from pyramidkv.quest import (
    build_quest_page_metadata,
    score_quest_pages,
    select_quest_pages,
    select_quest_tokens,
)


class QuestSelectionTest(unittest.TestCase):
    def test_page_metadata_tracks_min_and_max_per_page(self):
        keys = torch.tensor(
            [[[[1.0, -2.0], [3.0, 0.0], [-1.0, 4.0], [2.0, 5.0], [7.0, -3.0]]]]
        )

        page_min, page_max = build_quest_page_metadata(keys, page_size=2)

        self.assertEqual(tuple(page_min.shape), (1, 1, 3, 2))
        torch.testing.assert_close(page_min[0, 0, 0], torch.tensor([1.0, -2.0]))
        torch.testing.assert_close(page_max[0, 0, 1], torch.tensor([2.0, 5.0]))

    def test_scores_use_query_sign_to_choose_bounds(self):
        page_min = torch.tensor([[[[0.0, -5.0], [2.0, -1.0]]]])
        page_max = torch.tensor([[[[4.0, 1.0], [3.0, 6.0]]]])
        query = torch.tensor([[[2.0, -3.0]]])

        scores = score_quest_pages(query, page_min, page_max)

        torch.testing.assert_close(scores, torch.tensor([[[23.0, 9.0]]]))

    def test_select_pages_is_query_aware(self):
        keys = torch.tensor(
            [[[[4.0, 0.0], [3.0, 1.0], [-1.0, 5.0], [0.0, 6.0]]]]
        )
        x_query = torch.tensor([[[1.0, 0.0]]])
        y_query = torch.tensor([[[0.0, 1.0]]])

        x_pages = select_quest_pages(x_query, keys, page_size=2, page_budget=1)
        y_pages = select_quest_pages(y_query, keys, page_size=2, page_budget=1)

        torch.testing.assert_close(x_pages, torch.tensor([[[0]]]))
        torch.testing.assert_close(y_pages, torch.tensor([[[1]]]))

    def test_select_tokens_keeps_recent_window_and_exact_budget(self):
        keys = torch.tensor(
            [[[[4.0, 0.0], [3.0, 1.0], [-1.0, 5.0], [0.0, 6.0], [9.0, 9.0]]]]
        )
        query = torch.tensor([[[0.0, 1.0]]])

        tokens = select_quest_tokens(query, keys, page_size=2, token_budget=3, recent_size=1)

        torch.testing.assert_close(tokens, torch.tensor([[[2, 3, 4]]]))

    def test_select_tokens_does_not_return_partial_page_sentinel(self):
        keys = torch.tensor(
            [[[[1.0], [2.0], [3.0], [4.0], [100.0]]]]
        )
        query = torch.tensor([[[1.0]]])

        tokens = select_quest_tokens(query, keys, page_size=4, token_budget=4)

        self.assertEqual(tuple(tokens.shape), (1, 1, 4))
        self.assertLess(tokens.max().item(), keys.shape[2])


if __name__ == "__main__":
    unittest.main()
