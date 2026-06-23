import unittest

import torch

from pyramidkv.pyramidkv_utils import merge_kv


class MergeKVTest(unittest.TestCase):
    def test_pivot_merge_preserves_shape_and_recent_tokens(self):
        key_states = torch.eye(4).view(1, 1, 4, 4)
        value_states = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        indices = torch.tensor([[[0]]]).unsqueeze(-1).expand(-1, -1, -1, 4)

        merged_keys, merged_values = merge_kv(key_states, value_states, indices, window_size=1, merge="pivot")

        self.assertEqual(tuple(merged_keys.shape), (1, 1, 2, 4))
        self.assertEqual(tuple(merged_values.shape), (1, 1, 2, 4))
        torch.testing.assert_close(merged_keys[:, :, -1, :], key_states[:, :, -1, :])
        torch.testing.assert_close(merged_values[:, :, -1, :], value_states[:, :, -1, :])
        self.assertFalse(torch.equal(merged_keys[:, :, 0, :], key_states[:, :, 0, :]))

    def test_weighted_merge_supports_non_128_head_dim(self):
        key_states = torch.tensor(
            [[[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.9]]]]
        )
        value_states = torch.tensor(
            [[[[10.0, 0.0], [6.0, 2.0], [0.0, 10.0], [0.0, 8.0]]]]
        )
        indices = torch.tensor([[[0, 2]]])

        merged_keys, merged_values = merge_kv(key_states, value_states, indices, window_size=1, merge="weighted")

        self.assertEqual(tuple(merged_keys.shape), (1, 1, 3, 2))
        self.assertEqual(tuple(merged_values.shape), (1, 1, 3, 2))
        self.assertGreater(merged_keys[0, 0, 0, 1].item(), 0.0)
        self.assertGreater(merged_values[0, 0, 0, 1].item(), 0.0)
        torch.testing.assert_close(merged_values[:, :, -1, :], value_states[:, :, -1, :])


if __name__ == "__main__":
    unittest.main()
