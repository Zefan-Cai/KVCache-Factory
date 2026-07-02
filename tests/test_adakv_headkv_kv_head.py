"""AdaKV/HeadKV kv-head granularity: bit-identity and kv_head invariants.

Part 1 loads pyramidkv/pyramidkv_utils.py as it was at commit cdd65cf
(pre-AdaKV/HeadKV-kv_head) via `git show` + importlib and asserts torch.equal
outputs of AdaKVCluster.update_kv / HeadKVCluster.update_kv old-vs-new on
identical inputs, in both legacy call shapes:
  (a) MHA: query and key/value all at 4 heads;
  (b) query_head-mode GQA: key/value produced at 2 kv heads then pre-repeated
      to 8 heads with repeat_kv, exactly as the old model forwards passed them.
Both the returned flattened K/V tensors AND every metadata tensor/scalar built
by init_metadata are compared, since a metadata drift only explodes at decode
step 1.

Part 2 checks kv_head-mode invariants on unrepeated GQA inputs (kv_heads=2,
groups=4): stored-head-granular metadata, the AdaKV adaptive budget contract
(total ~= kv_heads * (base_capacity + window)), determinism, gqa_score_agg
mean-vs-max selection differences, the HeadKV per-query-head capacity table
group-MEAN reduction, the no-compress passthrough, and a pure-torch simulation
of the decode metadata increments.

GPU-only surfaces (DynamicCacheSplitHeadFlatten decode append via nvtx +
tiny_api_cuda, flash_attn_varlen_func decode attention) are guarded with skips
and belong to the Pluto GPU gate.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import torch

import pyramidkv.pyramidkv_utils as new_utils

REF_COMMIT = "cdd65cf"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BSZ = 1
HEAD_DIM = 16
WINDOW = 8
BASE_CAPACITY = 32
MAX_CAPACITY = BASE_CAPACITY + WINDOW  # 40
KERNEL = 7
POOLING = "maxpool"
FLOOR = 0.2
NUM_LAYERS = 4
LAYER_IDX = 1

# kv_head-mode shapes.
KV_HEADS = 2
GROUPS = 4
Q_HEADS = KV_HEADS * GROUPS  # 8
Q_LEN = 200

# Metadata attributes written by init_metadata in both old and new modules.
META_TENSORS = ("head_lens", "cu_headlens", "cu_klen", "layer_qlens",
                "cu_qlen", "cu_offset", "cu_head_offset")
META_SCALARS = ("klen_sum", "max_seqlen_k", "qlen_sum")


def _load_reference_module():
    source = subprocess.check_output(
        ["git", "show", f"{REF_COMMIT}:pyramidkv/pyramidkv_utils.py"], cwd=REPO_ROOT
    ).decode("utf-8")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_pyramidkv_utils_ref.py", delete=False
    )
    with tmp:
        tmp.write(source)
    name = f"pyramidkv_utils_ref_{REF_COMMIT}"
    spec = importlib.util.spec_from_file_location(name, tmp.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        os.unlink(tmp.name)
    return module


ref_utils = _load_reference_module()


def make_inputs(seed, num_q_heads, num_kv_heads, q_len, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(BSZ, num_q_heads, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    key = torch.randn(BSZ, num_kv_heads, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    value = torch.randn(BSZ, num_kv_heads, q_len, HEAD_DIM, generator=generator, dtype=dtype)
    return query, key, value


def make_head_capacity(num_layers, num_heads, seed=5):
    """Small synthetic per-QUERY-head capacity table, as run_longbench builds
    (integer, shape (num_layers, num_heads)); values stay well below
    q_len - window so the [..., :cap] slice never clamps in these tests."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(2, 21, (num_layers, num_heads), generator=generator, dtype=torch.int64)


def adakv_kwargs(**overrides):
    kwargs = dict(window_size=WINDOW, kernel_size=KERNEL, pooling=POOLING,
                  max_capacity_prompt=MAX_CAPACITY, floor=FLOOR, normalize=True,
                  layer_idx=LAYER_IDX, num_hidden_layers=NUM_LAYERS)
    kwargs.update(overrides)
    return kwargs


def headkv_kwargs(head_capacity, **overrides):
    kwargs = dict(window_size=WINDOW, kernel_size=KERNEL, pooling=POOLING,
                  max_capacity_prompt=MAX_CAPACITY, layer_idx=LAYER_IDX,
                  num_hidden_layers=NUM_LAYERS, head_capacity=head_capacity)
    kwargs.update(overrides)
    return kwargs


# (label, num_q_heads, num_kv_heads_before_repeat, repeat_factor)
INPUT_SHAPES = [
    ("mha", 4, 4, 1),
    ("gqa_pre_repeated", 8, 2, 4),
]
# Compression path (200) and the no-compress passthrough
# (30 - WINDOW = 22 < BASE_CAPACITY = 32).
Q_LENS = [200, 30]


class BitIdentityTest(unittest.TestCase):
    """Old-vs-new torch.equal on outputs AND metadata in the legacy shapes."""

    def _assert_identical(self, make_old, make_new):
        for label, q_heads, kv_heads, rep in INPUT_SHAPES:
            for q_len in Q_LENS:
                query, key, value = make_inputs(11 + q_len, q_heads, kv_heads, q_len)
                if rep > 1:
                    key = new_utils.repeat_kv(key, rep)
                    value = new_utils.repeat_kv(value, rep)

                old_cluster = make_old(q_heads)
                new_cluster = make_new(q_heads)
                old_k, old_v = old_cluster.update_kv(key.clone(), query.clone(), value.clone())
                new_k, new_v = new_cluster.update_kv(key.clone(), query.clone(), value.clone())

                msg = f"{label} q_len={q_len}"
                self.assertTrue(torch.equal(old_k, new_k), f"key mismatch: {msg}")
                self.assertTrue(torch.equal(old_v, new_v), f"value mismatch: {msg}")
                for attr in META_TENSORS:
                    self.assertTrue(
                        torch.equal(getattr(old_cluster, attr), getattr(new_cluster, attr)),
                        f"{attr} mismatch: {msg}")
                for attr in META_SCALARS:
                    self.assertEqual(getattr(old_cluster, attr), getattr(new_cluster, attr),
                                     f"{attr} mismatch: {msg}")

    def test_adakv(self):
        kwargs = adakv_kwargs()
        self._assert_identical(lambda h: ref_utils.AdaKVCluster(**kwargs),
                               lambda h: new_utils.AdaKVCluster(**kwargs))

    def test_adakv_no_normalize(self):
        kwargs = adakv_kwargs(normalize=False)
        self._assert_identical(lambda h: ref_utils.AdaKVCluster(**kwargs),
                               lambda h: new_utils.AdaKVCluster(**kwargs))

    def test_headkv(self):
        # The table is per QUERY head; in these legacy shapes stored heads ==
        # query heads, so the same table drives old and new identically.
        tables = {h: make_head_capacity(NUM_LAYERS, h) for h in (4, 8)}
        self._assert_identical(
            lambda h: ref_utils.HeadKVCluster(**headkv_kwargs(tables[h].clone())),
            lambda h: new_utils.HeadKVCluster(**headkv_kwargs(tables[h].clone())))


class AdaKVKvHeadInvariantsTest(unittest.TestCase):
    """kv_head mode: unrepeated K/V at KV_HEADS, query at Q_HEADS."""

    def _run(self, cluster, seed=0, q_len=Q_LEN):
        query, key, value = make_inputs(seed, Q_HEADS, KV_HEADS, q_len)
        k_out, v_out = cluster.update_kv(key.clone(), query.clone(), value.clone())
        return k_out, v_out, key, value, query

    def _check_metadata(self, cluster, k_out, v_out):
        self.assertEqual(k_out.dim(), 2)
        self.assertEqual(k_out.shape[1], HEAD_DIM)
        self.assertEqual(tuple(k_out.shape), tuple(v_out.shape))
        self.assertEqual(cluster.head_lens.shape[0], KV_HEADS)
        self.assertEqual(cluster.head_lens.sum().item(), cluster.klen_sum)
        self.assertEqual(k_out.shape[0], cluster.klen_sum)
        # cu_klen: KV_HEADS + 1 monotone offsets ending at klen_sum.
        self.assertEqual(cluster.cu_klen.shape[0], KV_HEADS + 1)
        self.assertEqual(cluster.cu_klen[0].item(), 0)
        self.assertEqual(cluster.cu_klen[-1].item(), cluster.klen_sum)
        self.assertTrue(bool((cluster.cu_klen[1:] >= cluster.cu_klen[:-1]).all()))
        expected_cu = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(cluster.head_lens, dim=0, dtype=torch.int32)])
        self.assertTrue(torch.equal(cluster.cu_klen, expected_cu))
        # One q-len-1 varlen sequence per stored head.
        arange = torch.arange(0, KV_HEADS + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(cluster.cu_qlen, arange))
        self.assertTrue(torch.equal(cluster.cu_offset, arange))
        self.assertEqual(cluster.max_seqlen_k, cluster.head_lens.max().item())

    def test_attn_score_rows_are_kv_heads(self):
        cluster = new_utils.AdaKVCluster(**adakv_kwargs())
        query, key, _ = make_inputs(3, Q_HEADS, KV_HEADS, Q_LEN)
        score = cluster.calcul_attn_sore(key, query)
        self.assertEqual(tuple(score.shape), (BSZ, KV_HEADS, Q_LEN - WINDOW))

    def test_compress_metadata_and_budget(self):
        cluster = new_utils.AdaKVCluster(**adakv_kwargs())
        k_out, v_out, key, value, _ = self._run(cluster)
        self._check_metadata(cluster, k_out, v_out)
        # Budget contract: per-stored-head budget preserved, so the total is
        # ~= KV_HEADS * (base_capacity + window); the floor mix rounds each
        # head's capacity, allowing +-1 per head.
        target = KV_HEADS * (BASE_CAPACITY + WINDOW)
        self.assertLessEqual(abs(cluster.klen_sum - target), KV_HEADS)
        # The floor keeps every head at >= floor_capacity + window (rounding -1).
        floor_capacity = int(BASE_CAPACITY * FLOOR)
        self.assertGreaterEqual(cluster.head_lens.min().item(),
                                floor_capacity + WINDOW - 1)
        # Each head's window (last WINDOW tokens) is kept verbatim at its
        # segment tail.
        offsets = cluster.cu_klen
        for h in range(KV_HEADS):
            seg_k = k_out[offsets[h]:offsets[h + 1]]
            self.assertTrue(torch.equal(seg_k[-WINDOW:], key[0, h, -WINDOW:, :]))
            seg_v = v_out[offsets[h]:offsets[h + 1]]
            self.assertTrue(torch.equal(seg_v[-WINDOW:], value[0, h, -WINDOW:, :]))

    def test_deterministic(self):
        outs = []
        for _ in range(2):
            cluster = new_utils.AdaKVCluster(**adakv_kwargs())
            k_out, v_out, *_ = self._run(cluster, seed=7)
            outs.append((k_out, v_out, cluster.head_lens.clone()))
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        self.assertTrue(torch.equal(outs[0][1], outs[1][1]))
        self.assertTrue(torch.equal(outs[0][2], outs[1][2]))

    def test_no_compress_passthrough(self):
        q_len = 30  # q_len - WINDOW = 22 < base_capacity = 32
        cluster = new_utils.AdaKVCluster(**adakv_kwargs())
        k_out, v_out, key, value, _ = self._run(cluster, seed=2, q_len=q_len)
        self.assertEqual(k_out.shape[0], q_len * KV_HEADS)
        self.assertTrue(torch.equal(k_out, key.reshape(-1, HEAD_DIM)))
        self.assertTrue(torch.equal(v_out, value.reshape(-1, HEAD_DIM)))
        self.assertTrue(torch.equal(
            cluster.head_lens, torch.full((KV_HEADS,), q_len, dtype=torch.int32)))
        self.assertEqual(cluster.klen_sum, q_len * KV_HEADS)
        self.assertEqual(cluster.max_seqlen_k, q_len)

    def test_agg_max_vs_mean_selection_differs(self):
        """A past token only ONE query head in a group attends to strongly must
        score (and select) differently under agg='max' vs agg='mean'."""
        q_len, window, base = 64, 4, 12
        generator = torch.Generator().manual_seed(3)
        query = 0.01 * torch.randn(BSZ, Q_HEADS, q_len, HEAD_DIM, generator=generator)
        key = 0.01 * torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator)
        value = torch.randn(BSZ, KV_HEADS, q_len, HEAD_DIM, generator=generator)
        target = 10
        # Only query head 0 (group 0 of kv head 0) spikes on past token 10.
        key[0, 0, target, :] = 0.0
        key[0, 0, target, 0] = 1.0
        query[0, 0, -window:, :] = 0.0
        query[0, 0, -window:, 0] = 50.0

        kwargs = adakv_kwargs(window_size=window, kernel_size=1,
                              max_capacity_prompt=base + window,
                              floor=0.0, normalize=False)
        results = {}
        for agg in ("mean", "max"):
            cluster = new_utils.AdaKVCluster(gqa_score_agg=agg, **kwargs)
            score = cluster.calcul_attn_sore(key.clone(), query.clone())
            k_out, _ = cluster.update_kv(key.clone(), query.clone(), value.clone())
            results[agg] = (score, k_out, cluster.head_lens.clone())
        score_mean, k_mean, _ = results["mean"]
        score_max, k_max, _ = results["max"]
        self.assertGreater(score_max[0, 0, target].item(),
                           score_mean[0, 0, target].item())
        self.assertFalse(torch.equal(score_mean, score_max))
        self.assertFalse(torch.equal(k_mean, k_max))
        # Under max, the outlier token must be among kv head 0's kept keys.
        row = results["max"][1][: results["max"][2][0].item()]
        self.assertTrue((row == key[0, 0, target, :]).all(dim=-1).any().item())


class HeadKVKvHeadInvariantsTest(unittest.TestCase):
    def test_capacity_table_group_mean_reduction(self):
        """Layer row [8,4,6,2, 12,2,4,6] over 8 query heads must reduce to
        per-kv-head capacities [mean(8,4,6,2), mean(12,2,4,6)] = [5, 6]."""
        table = make_head_capacity(NUM_LAYERS, Q_HEADS)
        table[LAYER_IDX] = torch.tensor([8, 4, 6, 2, 12, 2, 4, 6])
        cluster = new_utils.HeadKVCluster(**headkv_kwargs(table))
        query, key, value = make_inputs(9, Q_HEADS, KV_HEADS, Q_LEN)
        k_out, v_out = cluster.update_kv(key.clone(), query.clone(), value.clone())

        expected_lens = torch.tensor([5 + WINDOW, 6 + WINDOW], dtype=torch.int32)
        self.assertTrue(torch.equal(cluster.head_lens, expected_lens))
        self.assertEqual(cluster.klen_sum, expected_lens.sum().item())
        self.assertEqual(k_out.shape[0], cluster.klen_sum)
        self.assertTrue(torch.equal(
            cluster.cu_klen, torch.tensor([0, 13, 27], dtype=torch.int32)))
        arange = torch.arange(0, KV_HEADS + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(cluster.cu_qlen, arange))
        self.assertTrue(torch.equal(cluster.cu_offset, arange))
        self.assertEqual(cluster.max_seqlen_k, expected_lens.max().item())
        # Windows kept verbatim per stored head.
        offsets = cluster.cu_klen
        for h in range(KV_HEADS):
            self.assertTrue(torch.equal(
                k_out[offsets[h]:offsets[h + 1]][-WINDOW:], key[0, h, -WINDOW:, :]))
            self.assertTrue(torch.equal(
                v_out[offsets[h]:offsets[h + 1]][-WINDOW:], value[0, h, -WINDOW:, :]))

    def test_mean_reduction_rounds(self):
        # [3,4,4,4] -> mean 3.75 -> round 4; [2,2,2,3] -> 2.25 -> 2.
        table = make_head_capacity(NUM_LAYERS, Q_HEADS)
        table[LAYER_IDX] = torch.tensor([3, 4, 4, 4, 2, 2, 2, 3])
        cluster = new_utils.HeadKVCluster(**headkv_kwargs(table))
        query, key, value = make_inputs(10, Q_HEADS, KV_HEADS, Q_LEN)
        cluster.update_kv(key, query, value)
        self.assertTrue(torch.equal(
            cluster.head_lens,
            torch.tensor([4 + WINDOW, 2 + WINDOW], dtype=torch.int32)))

    def test_deterministic(self):
        table = make_head_capacity(NUM_LAYERS, Q_HEADS)
        query, key, value = make_inputs(12, Q_HEADS, KV_HEADS, Q_LEN)
        outs = []
        for _ in range(2):
            cluster = new_utils.HeadKVCluster(**headkv_kwargs(table.clone()))
            outs.append(cluster.update_kv(key.clone(), query.clone(), value.clone()))
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        self.assertTrue(torch.equal(outs[0][1], outs[1][1]))

    def test_no_compress_passthrough(self):
        q_len = 30
        table = make_head_capacity(NUM_LAYERS, Q_HEADS)
        cluster = new_utils.HeadKVCluster(**headkv_kwargs(table))
        query, key, value = make_inputs(13, Q_HEADS, KV_HEADS, q_len)
        k_out, v_out = cluster.update_kv(key.clone(), query.clone(), value.clone())
        self.assertEqual(k_out.shape[0], q_len * KV_HEADS)
        self.assertTrue(torch.equal(k_out, key.reshape(-1, HEAD_DIM)))
        self.assertTrue(torch.equal(v_out, value.reshape(-1, HEAD_DIM)))

    def test_attn_score_rows_are_kv_heads(self):
        table = make_head_capacity(NUM_LAYERS, Q_HEADS)
        cluster = new_utils.HeadKVCluster(**headkv_kwargs(table))
        query, key, _ = make_inputs(14, Q_HEADS, KV_HEADS, Q_LEN)
        score = cluster.calcul_attn_sore(key, query)
        self.assertEqual(tuple(score.shape), (BSZ, KV_HEADS, Q_LEN - WINDOW))


class DecodeMetadataSimulationTest(unittest.TestCase):
    """Pure-torch simulation of the per-step decode metadata mutation done by
    the model forwards (klen_sum += num_cached_heads, max_seqlen_k += 1,
    cu_klen += cu_offset, head_lens += 1), validated against a from-scratch
    rebuild of the varlen metadata over the incremented lengths. The actual
    cache append (nvtx + tiny_api_cuda) is CUDA-only and belongs to the GPU
    gate."""

    def test_increments_match_rebuild(self):
        cluster = new_utils.AdaKVCluster(**adakv_kwargs())
        query, key, value = make_inputs(21, Q_HEADS, KV_HEADS, Q_LEN)
        cluster.update_kv(key, query, value)
        num_cached_heads = cluster.head_lens.shape[0]
        self.assertEqual(num_cached_heads, KV_HEADS)

        base_lens = cluster.head_lens.clone()
        for step in range(1, 4):
            # Mirror the decode-branch mutation in the model forwards.
            cluster.klen_sum += num_cached_heads
            cluster.max_seqlen_k += 1
            cluster.cu_klen += cluster.cu_offset
            cluster.head_lens += 1

            expected_lens = base_lens + step
            self.assertTrue(torch.equal(cluster.head_lens, expected_lens), f"step {step}")
            self.assertEqual(cluster.klen_sum, expected_lens.sum().item(), f"step {step}")
            expected_cu = torch.cat([
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(expected_lens, dim=0, dtype=torch.int32)])
            self.assertTrue(torch.equal(cluster.cu_klen, expected_cu), f"step {step}")
            self.assertEqual(cluster.max_seqlen_k, expected_lens.max().item(), f"step {step}")


@unittest.skipUnless(torch.cuda.is_available(),
                     "kv_head decode cache append needs CUDA (nvtx + tiny_api_cuda); "
                     "covered by the Pluto GPU gate (task #8)")
class GpuOnlySurfacesTest(unittest.TestCase):
    def test_flatten_cache_decode_append(self):
        try:
            import nvtx  # noqa: F401
            import tiny_api_cuda  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"GPU decode deps unavailable: {exc}")
        device = "cuda"
        cluster = new_utils.AdaKVCluster(**adakv_kwargs())
        query, key, value = make_inputs(30, Q_HEADS, KV_HEADS, Q_LEN)
        k_out, v_out = cluster.update_kv(key.to(device), query.to(device), value.to(device))
        cache = new_utils.DynamicCacheSplitHeadFlatten()
        cache.update(k_out, v_out, 0)
        before = cache.key_cache[0].shape[0]
        new_k = torch.randn(1, KV_HEADS, 1, HEAD_DIM, device=device)
        new_v = torch.randn(1, KV_HEADS, 1, HEAD_DIM, device=device)
        kwargs = {"head_lens": cluster.head_lens, "cu_klen": cluster.cu_klen}
        cache.update(new_k, new_v, 0, kwargs)
        self.assertEqual(cache.key_cache[0].shape[0], before + KV_HEADS)


if __name__ == "__main__":
    unittest.main()
