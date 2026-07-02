# GQA KV-Cache Layout Audit (issue #49)

Evidence date: 2026-07-02.

This note answers the GQA cache-layout question in
[issue #49](https://github.com/Zefan-Cai/KVCache-Factory/issues/49) and records a
concrete, validation-gated migration plan. It is an audit + plan, not a landed
change: the correct refactor alters token-selection semantics and cannot be
verified without full benchmark reruns, so it must not be applied blindly.

## Current behavior

Every compression forward stores the KV cache at **query-head granularity**, not
KV-head granularity. The common pattern (e.g. `pyramidkv/llama_model.py:158`,
`:167`, `:168`) is:

```python
key_states   = repeat_kv(key_states, self.num_key_value_groups)   # KV heads -> query heads
value_states = repeat_kv(value_states, self.num_key_value_groups)
...
key_states_compress, value_states_compress = self.kv_cluster.update_kv(
    key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
```

So `repeat_kv` runs **before** compression, and the expanded, compressed tensor
is what gets written into the cache.

Scope of the pattern (grepped 2026-07-02):

| Location | `repeat_kv`-before-compress sites | `kv_cluster.update_kv` call sites |
| --- | --- | --- |
| `pyramidkv/llama_model.py` | 20 | 20 |
| `pyramidkv/mistral_model.py` | 20 | 20 |

Cluster classes with an `update_kv` that assumes query-head-granularity input
(`pyramidkv/pyramidkv_utils.py`): `PyramidKVCluster`, `SnapKVCluster`,
`L2NormCluster`, `CAMKVCluster`, `H2OKVCluster`, `StreamingLLMKVCluster`,
`AdaKVCluster`, `HeadKVCluster` — 8 classes.

## Memory implication

For a GQA model the stored cache is `num_key_value_groups` times larger than a
KV-head-granularity cache. For Meta-Llama-3-8B (32 query heads, 8 KV heads,
`num_key_value_groups = 4`), the compressed KV cache is **~4x** the size an
efficient GQA cache would use. This negates GQA's main KV-cache memory benefit:
the reported `max_capacity_prompt` budget is spent per query head rather than per
KV head. The reporter in #49 is correct on this point.

## Why this is not a mechanical refactor

Moving compression before `repeat_kv` (score/select at KV-head granularity, then
`repeat_kv` only for attention compute) **changes which tokens are kept**, so it
is a semantics change, not a behaviour-preserving refactor:

- Selection is currently **per query head**. `SnapKVCluster.update_kv`
  (`pyramidkv/pyramidkv_utils.py:378`) does `attn_cache.topk(...)` producing a
  distinct kept-token set for each of the 32 query heads; `H2OKVCluster`
  (`:593`) does the same on accumulated scores.
- At KV-head granularity there is only one cache slot per KV head, shared by its
  `num_key_value_groups` query heads. The 4 query heads in a group generally
  rank different tokens, so a single kept set per KV head requires **reducing**
  the 4 heads' scores (sum / mean / max) before top-k. That reduction is a new
  modeling choice that changes the retained tokens and therefore the model
  output and downstream accuracy.
- `AdaKVCluster` / `HeadKVCluster` add per-head *budgets* on top of per-head
  selection, so their semantics under head grouping need separate design.

In short: the "efficient" layout is not free — it trades a documented ~4x memory
win for a change in the selection algorithm that every method must re-justify.

## Migration plan (validation-gated)

Do not change all 40 call sites at once. Land incrementally, each step behind a
benchmark gate, because correctness can only be shown on GPU (LongBench / RULER),
not by unit tests alone.

1. **Pin current behavior.** Record baseline LongBench/RULER scores and peak KV
   memory for SnapKV + Llama-3-8B on `main` (blocked today: the MS H100 fleet is
   fully occupied / the N8 cluster is offline — see the #46 reproduction task).
2. **Prototype one method, one model.** Add a KV-head-granularity path for
   `SnapKVCluster` on Llama only, behind an explicit flag
   (e.g. `--kv_cache_granularity kv_head`, default `query_head`). Implement an
   explicit group-score reduction (start with `mean`, expose the reduction op)
   before top-k, and `repeat_kv` at attention-compute time instead of before
   `update_kv`.
3. **Prove parity/tradeoff.** Compare against the pinned baseline: expect ~4x
   lower KV memory and quantify the accuracy delta from group reduction. Keep the
   flag off by default until the delta is acceptable.
4. **Generalize.** Only after (3), extend to the other clusters and to Mistral,
   reusing the same reduction contract. `AdaKV`/`HeadKV` need per-head-budget
   redesign and should come last.

Until step 1 can run, the query-head-granularity layout stays the default and
this issue remains open. This matches the maintainer's note on #49 that a correct
refactor needs a wider audit rather than a risky local change.
