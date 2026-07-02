"""Default-mode bit-identity of the flag-gated GQA refactor.

Loads pyramidkv/pyramidkv_utils.py as it was at commit 76b07dc (pre-refactor)
via `git show` + importlib and asserts torch.equal update_kv outputs against
the current module for every in-scope cluster, on identical inputs, in both
legacy call shapes:
  (a) MHA: query and key/value all at 4 heads;
  (b) query_head-mode GQA: key/value produced at 2 kv heads then pre-repeated
      to 8 heads with repeat_kv, exactly as the old model forwards passed them.
"""

import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest

import torch

import pyramidkv.pyramidkv_utils as new_utils

REF_COMMIT = "76b07dc"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BSZ = 1
HEAD_DIM = 16
WINDOW = 8
CAPACITY = 32
KERNEL = 7
POOLING = "maxpool"


def _load_reference_module():
    source = subprocess.check_output(
        ["git", "show", f"{REF_COMMIT}:pyramidkv/pyramidkv_utils.py"], cwd=REPO_ROOT
    ).decode("utf-8")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_pyramidkv_utils_ref.py", delete=False
    )
    with tmp:
        tmp.write(source)
    name = "pyramidkv_utils_ref_76b07dc"
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


# (label, num_q_heads, num_kv_heads_before_repeat, repeat_factor)
INPUT_SHAPES = [
    ("mha", 4, 4, 1),
    ("gqa_pre_repeated", 8, 2, 4),
]
# q_len cases: long prompt (compression), short prompt (passthrough), and one
# that lands in PyramidKV's middle branch for CAPACITY=32 (32 <= 40 < 48).
Q_LENS = [200, 40, CAPACITY - 5]


class BitIdentityTest(unittest.TestCase):
    def _assert_identical(self, make_old, make_new, seed_rng=False, q_lens=Q_LENS, merge=None):
        for label, q_heads, kv_heads, rep in INPUT_SHAPES:
            for q_len in q_lens:
                query, key, value = make_inputs(11 + q_len, q_heads, kv_heads, q_len)
                if rep > 1:
                    key = new_utils.repeat_kv(key, rep)
                    value = new_utils.repeat_kv(value, rep)
                num_kv_groups = rep  # what the old forwards passed for GQA models

                if seed_rng:
                    torch.manual_seed(99)  # CAM's bernoulli uses the global RNG
                old_k, old_v = make_old().update_kv(
                    key.clone(), query.clone(), value.clone(), None, num_kv_groups
                )
                if seed_rng:
                    torch.manual_seed(99)
                new_k, new_v = make_new().update_kv(
                    key.clone(), query.clone(), value.clone(), None, num_kv_groups
                )
                msg = f"{label} q_len={q_len} merge={merge}"
                self.assertTrue(torch.equal(old_k, new_k), f"key mismatch: {msg}")
                self.assertTrue(torch.equal(old_v, new_v), f"value mismatch: {msg}")

    def test_snapkv(self):
        kwargs = dict(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                      kernel_size=KERNEL, pooling=POOLING)
        self._assert_identical(lambda: ref_utils.SnapKVCluster(**kwargs),
                               lambda: new_utils.SnapKVCluster(**kwargs))

    def test_snapkv_merge_pivot(self):
        kwargs = dict(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                      kernel_size=KERNEL, pooling=POOLING, merge="pivot")
        self._assert_identical(lambda: ref_utils.SnapKVCluster(**kwargs),
                               lambda: new_utils.SnapKVCluster(**kwargs), merge="pivot")

    def test_pyramidkv(self):
        # layer_idx in the middle of the schedule; q_len=40 exercises the middle
        # branch, 200 the long branch, 27 the passthrough.
        kwargs = dict(num_hidden_layers=4, layer_idx=1, window_size=WINDOW,
                      max_capacity_prompt=CAPACITY, kernel_size=KERNEL, pooling=POOLING)
        self._assert_identical(lambda: ref_utils.PyramidKVCluster(**kwargs),
                               lambda: new_utils.PyramidKVCluster(**kwargs))

    def test_h2o(self):
        kwargs = dict(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                      kernel_size=KERNEL, pooling=POOLING)
        self._assert_identical(lambda: ref_utils.H2OKVCluster(**kwargs),
                               lambda: new_utils.H2OKVCluster(**kwargs))

    def test_streamingllm(self):
        kwargs = dict(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                      kernel_size=KERNEL, pooling=POOLING)
        self._assert_identical(lambda: ref_utils.StreamingLLMKVCluster(**kwargs),
                               lambda: new_utils.StreamingLLMKVCluster(**kwargs))

    def test_cam(self):
        kwargs = dict(window_size=WINDOW, max_capacity_prompt=CAPACITY,
                      kernel_size=KERNEL, pooling=POOLING)
        self._assert_identical(lambda: ref_utils.CAMKVCluster(**kwargs),
                               lambda: new_utils.CAMKVCluster(**kwargs), seed_rng=True)

    def test_l2norm(self):
        kwargs = dict(max_capacity_prompt=CAPACITY, layer_idx=1, skip_layers=[])
        self._assert_identical(lambda: ref_utils.L2NormCluster(**kwargs),
                               lambda: new_utils.L2NormCluster(**kwargs))


if __name__ == "__main__":
    unittest.main()
