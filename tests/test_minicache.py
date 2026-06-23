import unittest

import torch

from pyramidkv.minicache import (
    compress_minicache_pair,
    minicache_slerp,
    restore_minicache_pair,
    select_minicache_retention_indices,
)


class MiniCacheTest(unittest.TestCase):
    def test_slerp_preserves_unit_direction_and_magnitudes(self):
        previous = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
        current = torch.tensor([[[[0.0, 3.0], [4.0, 0.0]]]])

        shared, current_mag, previous_mag, angle = minicache_slerp(current, previous, interpolation=0.5)

        torch.testing.assert_close(shared.norm(dim=-1), torch.ones_like(angle))
        torch.testing.assert_close(current_mag.squeeze(-1), current.norm(dim=-1))
        torch.testing.assert_close(previous_mag.squeeze(-1), previous.norm(dim=-1))
        torch.testing.assert_close(angle, torch.full_like(angle, torch.pi / 2), atol=2e-3, rtol=0)

    def test_retention_keeps_largest_angular_distance_indices(self):
        distances = torch.tensor([[[0.05, 0.9, 0.2, 0.8, 0.1]]])

        indices = select_minicache_retention_indices(distances, retention_count=2)

        torch.testing.assert_close(indices, torch.tensor([[[1, 3]]]))

    def test_compress_restore_keeps_retained_tokens_exact(self):
        previous = torch.tensor(
            [[[[1.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 3.0]]]]
        )
        current = torch.tensor(
            [[[[0.8, 0.2], [2.0, 0.0], [0.0, 2.0], [0.0, 4.0]]]]
        )

        (
            shared,
            current_mag,
            previous_mag,
            distances,
            current_retained,
            previous_retained,
            retained_indices,
        ) = compress_minicache_pair(current, previous, retention_count=1)
        restored_current, restored_previous = restore_minicache_pair(
            shared,
            current_mag,
            previous_mag,
            current_retained=current_retained,
            previous_retained=previous_retained,
            retained_indices=retained_indices,
        )

        self.assertEqual(tuple(distances.shape), (1, 1, 4))
        self.assertEqual(tuple(retained_indices.shape), (1, 1, 1))
        torch.testing.assert_close(
            restored_current.gather(2, retained_indices.unsqueeze(-1).expand(-1, -1, -1, 2)),
            current_retained,
        )
        torch.testing.assert_close(
            restored_previous.gather(2, retained_indices.unsqueeze(-1).expand(-1, -1, -1, 2)),
            previous_retained,
        )
        torch.testing.assert_close(restored_current.norm(dim=-1), current.norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(restored_previous.norm(dim=-1), previous.norm(dim=-1), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
