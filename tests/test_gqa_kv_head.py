import math
import unittest

import torch
import torch.nn.functional as F

from pyramidkv.pyramidkv_utils import (
    CAMKVCluster,
    H2OKVCluster,
    L2NormCluster,
    PyramidKVCluster,
    SnapKVCluster,
    StreamingLLMKVCluster,
    _reduce_group_scores,
    repeat_kv,
)

BSZ = 1
KV_HEADS = 2
GROUPS = 4
Q_HEADS = KV_HEADS * GROUPS  # 8
Q_LEN = 200
HEAD_DIM = 16
WINDOW = 8
CAPACITY = 32
KERNEL = 7
POOLING = "maxpool"


def make_inputs(seed=0, q_len=Q_LEN, dtype=torch.float32):
    """Unrepeated GQA tensors: query at Q_HEADS, key/value at KV_HEADS."""
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(BSZ, Q_HEADS, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    key = torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    value = torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    return query, key, value


def naive_snapkv_kv_head(query, key, value, groups, window_size, kernel_size, pooling, capacity, agg):
    """Independent reference for the SnapKV kv_head selection: transient repeat
    -> window scores with the SnapKV causal mask -> fp32 softmax -> row-sum ->
    (bsz, kv_heads, groups, len) group reduction -> pool1d -> topk -> gather on
    the unrepeated tensors."""
    head_dim = query.shape[-1]
    key_rep = repeat_kv(key, groups)
    attn = torch.matmul(query[..., -window_size:, :], key_rep.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full((window_size, window_size), torch.finfo(attn.dtype).min)
    mask_cond = torch.arange(window_size)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(window_size, 1), 0)
    attn[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    scores = attn[:, :, -window_size:, :-window_size].sum(dim=-2)
    bsz, q_heads, past_len = scores.shape
    grouped = scores.reshape(bsz, q_heads // groups, groups, past_len)
    if agg == "mean":
        scores = grouped.mean(dim=2)
    elif agg == "max":
        scores = grouped.amax(dim=2)
    elif agg == "sum":
        scores = grouped.sum(dim=2)
    else:
        raise ValueError(agg)
    if pooling == "maxpool":
        cache = F.max_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    else:
        cache = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    indices = cache.topk(capacity - window_size, dim=-1).indices
    idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past = key[:, :, :-window_size, :].gather(dim=2, index=idx)
    v_past = value[:, :, :-window_size, :].gather(dim=2, index=idx)
    k_out = torch.cat([k_past, key[:, :, -window_size:, :]], dim=2)
    v_out = torch.cat([v_past, value[:, :, -window_size:, :]], dim=2)
    return k_out, v_out, indices


class KVHeadModeShapeTest(unittest.TestCase):
    """kv_head mode invariants for every in-scope cluster on unrepeated GQA inputs."""

    def _check_common(self, key_out, value_out, key_in, value_in, kept_cap, check_window=True):
        self.assertEqual(key_out.shape[0], BSZ)
        self.assertEqual(key_out.shape[1], KV_HEADS)
        self.assertEqual(key_out.shape[3], HEAD_DIM)
        self.assertEqual(tuple(key_out.shape), tuple(value_out.shape))
        self.assertLessEqual(key_out.shape[2], kept_cap)
        if check_window:
            torch.testing.assert_close(key_out[:, :, -WINDOW:, :], key_in[:, :, -WINDOW:, :])
            torch.testing.assert_close(value_out[:, :, -WINDOW:, :], value_in[:, :, -WINDOW:, :])

    def _run_twice(self, make_cluster, seed_rng=False):
        """Run update_kv twice on cloned identical inputs; return both results."""
        query, key, value = make_inputs()
        results = []
        for _ in range(2):
            if seed_rng:
                torch.manual_seed(1234)  # CAM draws from the global RNG (bernoulli)
            cluster = make_cluster()
            k_out, v_out = cluster.update_kv(
                key.clone(), query.clone(), value.clone(), attention_mask=None, num_key_value_groups=GROUPS
            )
            results.append((k_out, v_out))
        (k1, v1), (k2, v2) = results
        self.assertTrue(torch.equal(k1, k2), "kv_head output not deterministic")
        self.assertTrue(torch.equal(v1, v2), "kv_head output not deterministic")
        return k1, v1, key, value

    def test_snapkv_kv_head(self):
        k_out, v_out, key, value = self._run_twice(
            lambda: SnapKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                                  kernel_size=KERNEL, pooling=POOLING)
        )
        self._check_common(k_out, v_out, key, value, CAPACITY)
        self.assertEqual(k_out.shape[2], CAPACITY)

    def test_pyramidkv_kv_head_middle_branch(self):
        # max_capacity_prompt=128, window=8 -> q_len=200 < (128-8)*2=240 takes the
        # middle branch, which keeps (max_capacity_prompt - window) + window tokens.
        cap = 128
        k_out, v_out, key, value = self._run_twice(
            lambda: PyramidKVCluster(num_hidden_layers=4, layer_idx=1, window_size=WINDOW,
                                     max_capacity_prompt=cap, kernel_size=KERNEL, pooling=POOLING)
        )
        self._check_common(k_out, v_out, key, value, cap)
        self.assertEqual(k_out.shape[2], cap)

    def test_pyramidkv_kv_head_matches_query_head_kept_length(self):
        # The per-layer schedule must be identical in both modes (long branch here).
        query, key, value = make_inputs()
        cluster_kv = PyramidKVCluster(num_hidden_layers=4, layer_idx=1, window_size=WINDOW,
                                      max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING)
        k_kv, _ = cluster_kv.update_kv(key.clone(), query.clone(), value.clone(), None, GROUPS)
        cluster_q = PyramidKVCluster(num_hidden_layers=4, layer_idx=1, window_size=WINDOW,
                                     max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING)
        key_rep, value_rep = repeat_kv(key, GROUPS), repeat_kv(value, GROUPS)
        k_q, _ = cluster_q.update_kv(key_rep, query.clone(), value_rep, None, 1)
        self.assertEqual(k_kv.shape[2], k_q.shape[2])
        self.assertEqual(k_kv.shape[1], KV_HEADS)
        self.assertEqual(k_q.shape[1], Q_HEADS)

    def test_h2o_kv_head(self):
        k_out, v_out, key, value = self._run_twice(
            lambda: H2OKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                                 kernel_size=KERNEL, pooling=POOLING)
        )
        self._check_common(k_out, v_out, key, value, CAPACITY)
        self.assertEqual(k_out.shape[2], CAPACITY)

    def test_cam_kv_head(self):
        k_out, v_out, key, value = self._run_twice(
            lambda: CAMKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                                 kernel_size=KERNEL, pooling=POOLING),
            seed_rng=True,
        )
        # CAM merges dropped V rows into neighbours, so only K keeps the raw window.
        self._check_common(k_out, v_out, key, value, CAPACITY, check_window=False)
        torch.testing.assert_close(k_out[:, :, -WINDOW:, :], key[:, :, -WINDOW:, :])
        self.assertEqual(k_out.shape[2], CAPACITY)

    def test_streamingllm_kv_head_exact_sinks_plus_recent(self):
        k_out, v_out, key, value = self._run_twice(
            lambda: StreamingLLMKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                                          kernel_size=KERNEL, pooling=POOLING)
        )
        self._check_common(k_out, v_out, key, value, CAPACITY)
        sinks = CAPACITY - WINDOW
        self.assertTrue(torch.equal(k_out, torch.cat([key[:, :, :sinks, :], key[:, :, -WINDOW:, :]], dim=2)))
        self.assertTrue(torch.equal(v_out, torch.cat([value[:, :, :sinks, :], value[:, :, -WINDOW:, :]], dim=2)))

    def test_l2norm_kv_head(self):
        # L2Norm keeps the lowest-key-norm tokens; it has no observation window.
        k_out, v_out, key, value = self._run_twice(
            lambda: L2NormCluster(max_capacity_prompt=CAPACITY, layer_idx=1)
        )
        self._check_common(k_out, v_out, key, value, CAPACITY, check_window=False)
        self.assertEqual(k_out.shape[2], CAPACITY)
        norms = torch.norm(key, p=2, dim=-1)
        expected_idx = norms.argsort(dim=-1)[:, :, :CAPACITY]
        expected_k = key.gather(dim=2, index=expected_idx.unsqueeze(-1).expand(-1, -1, -1, HEAD_DIM))
        self.assertTrue(torch.equal(k_out, expected_k))


class SnapKVNaiveReferenceTest(unittest.TestCase):
    def test_selection_matches_naive_reference(self):
        query, key, value = make_inputs(seed=7)
        for agg in ("mean", "max", "sum"):
            cluster = SnapKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                                    kernel_size=KERNEL, pooling=POOLING, gqa_score_agg=agg)
            k_out, v_out = cluster.update_kv(key.clone(), query.clone(), value.clone(), None, GROUPS)
            k_ref, v_ref, _ = naive_snapkv_kv_head(
                query, key, value, GROUPS, WINDOW, KERNEL, POOLING, CAPACITY, agg)
            self.assertTrue(torch.equal(k_out, k_ref), f"key selection mismatch for agg={agg}")
            self.assertTrue(torch.equal(v_out, v_ref), f"value selection mismatch for agg={agg}")


class ScoreAggregationTest(unittest.TestCase):
    def test_reduce_group_scores_exact_values(self):
        # (bsz=1, q_heads=4, len=2) with groups=2 -> kv_heads=2.
        scores = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [10.0, 0.0], [20.0, 4.0]]])
        mean = _reduce_group_scores(scores, 2, "mean")
        self.assertTrue(torch.equal(mean, torch.tensor([[[2.0, 4.0], [15.0, 2.0]]])))
        amax = _reduce_group_scores(scores, 2, "max")
        self.assertTrue(torch.equal(amax, torch.tensor([[[3.0, 6.0], [20.0, 4.0]]])))
        total = _reduce_group_scores(scores, 2, "sum")
        self.assertTrue(torch.equal(total, torch.tensor([[[4.0, 8.0], [30.0, 4.0]]])))

    def test_reduce_group_scores_groups_one_is_identity(self):
        scores = torch.randn(1, 4, 6)
        self.assertTrue(torch.equal(_reduce_group_scores(scores, 1, "mean"), scores))

    def test_reduce_group_scores_rejects_unknown_agg(self):
        with self.assertRaises(ValueError):
            _reduce_group_scores(torch.randn(1, 4, 6), 2, "median")

    def test_max_agg_selects_outlier_head_token(self):
        """A token that one query head in the group attends to strongly must be
        kept under agg='max' but diluted away under agg='mean'."""
        q_len = 64
        window, capacity, kernel = 4, 12, 1  # kernel=1 -> no pooling smear
        generator = torch.Generator().manual_seed(3)
        query = 0.01 * torch.randn(BSZ, Q_HEADS, q_len, HEAD_DIM, generator=generator)
        key = 0.01 * torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator)
        value = torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator)
        # Make ONLY query head 0 (group 0 of kv head 0) spike on past token 10.
        target = 10
        key[0, 0, target, :] = 0.0
        key[0, 0, target, 0] = 1.0
        query[0, 0, -window:, :] = 0.0
        query[0, 0, -window:, 0] = 50.0

        outputs = {}
        for agg in ("mean", "max"):
            cluster = SnapKVCluster(window_size=window, max_capacity_prompt=capacity,
                                    kernel_size=kernel, pooling=POOLING, gqa_score_agg=agg)
            k_out, _ = cluster.update_kv(key.clone(), query.clone(), value.clone(), None, GROUPS)
            k_ref, _, indices = naive_snapkv_kv_head(
                query, key, value, GROUPS, window, kernel, POOLING, capacity, agg)
            self.assertTrue(torch.equal(k_out, k_ref))
            outputs[agg] = indices
        self.assertIn(target, outputs["max"][0, 0].tolist())
        self.assertFalse(torch.equal(outputs["mean"], outputs["max"]))


class NoCompressionBelowCapacityTest(unittest.TestCase):
    def test_short_prompt_passthrough_kv_head(self):
        query, key, value = make_inputs(seed=2, q_len=CAPACITY - 1)
        for cluster in (
            SnapKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING),
            H2OKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING),
            StreamingLLMKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING),
            CAMKVCluster(window_size=WINDOW, max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING),
            L2NormCluster(max_capacity_prompt=CAPACITY, layer_idx=1),
            PyramidKVCluster(num_hidden_layers=4, layer_idx=1, window_size=WINDOW,
                             max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING),
        ):
            k_out, v_out = cluster.update_kv(key.clone(), query.clone(), value.clone(), None, GROUPS)
            self.assertTrue(torch.equal(k_out, key), type(cluster).__name__)
            self.assertTrue(torch.equal(v_out, value), type(cluster).__name__)


if __name__ == "__main__":
    unittest.main()
