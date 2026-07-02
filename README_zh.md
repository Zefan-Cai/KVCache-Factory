# KVCache-Factory

KVCache-Factory 是一个面向长上下文 LLM 推理的 KV cache 方法集合，覆盖压缩、检索、合并和量化等方向。项目从 PyramidKV 扩展而来，目前把多种 KV cache baseline 放到统一的评测入口下。

## 当前支持

- 压缩/检索：`PyramidKV`、`SnapKV`、`Quest`、`NACL`、`Scissorhands`、`MiniCache`、`H2O`、`StreamingLLM`、`CAM`、`L2Norm`、`AdaKV`、`HeadKV`、`ThinK`
- 合同工具：`Quest` 风格 query-aware page/token selector、`NACL` 风格 proxy/random eviction selector、`Scissorhands` 风格 persistence-of-importance selector、`MiniCache` 风格 cross-layer SLERP merge/restore、`LOOK-M` 风格 `pivot` merge、`KVMerger` 风格 `weighted` merge
- 量化：`--quant_method kivi`、`--quant_method kvquant`，其中 KIVI 默认 key axis 为 `1`、value axis 为 `0`
- 模型路径：主要支持 Llama 和 Mistral 的 Hugging Face attention monkeypatch
- 评测：LongBench、Needle-in-a-haystack、RULER、单 prompt latency/memory benchmark

## 快速开始

```bash
git clone https://github.com/Zefan-Cai/KVCache-Factory.git
cd KVCache-Factory
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

LongBench 示例：

```bash
python3 run_longbench.py \
  --method pyramidkv \
  --model_path /path/to/Llama-3-8B-Instruct \
  --max_capacity_prompts 128 \
  --attn_implementation flash_attention_2 \
  --save_dir ./results_long_bench \
  --use_cache True
```

快速开始使用 `128` 的 KV cache 预算；PyramidKV 论文报告的是 `128` 和 `2048` 两档预算。

常用新参数：

- `--datasets`：逗号分隔的 LongBench 数据集列表（默认评测全部 16 个数据集）
- `--dtype`：`float16`（默认）、`bfloat16` 或 `auto`
- `--kv_cache_granularity`：`query_head`（默认，旧布局）或 `kv_head`（GQA 高效布局，支持 `snapkv`、`pyramidkv`、`h2o`、`streamingllm`、`cam`、`l2norm`，以及 `adakv`/`headkv`——GPU 验证待完成）；RULER（`run_ruler.py`）和大海捞针（`run_needle_in_haystack.py`）脚本也接受相同参数，详见 `docs/gqa_cache_layout.md`
- `--gqa_score_agg`：`mean`（默认）、`max` 或 `sum`，`kv_head` 布局下每个 KV head 的打分聚合方式

## 复现备注

- Llama-3 的 LongBench 运行现在使用官方 LongBench 的 chat 模板，并同时在 `<|eot_id|>` 与 `<|end_of_text|>` 两个终止符上停止；早期版本只用单一 EOS 且没有 Llama-3 chat 包裹，会压低分数（issue #46），因此与早期版本的分数不可直接比较。
- `transformers` 固定为 `4.44.2`。
- 量化（`--quant_method`）与合并（`--merge`）默认关闭。
- 每次 LongBench 运行会在结果目录写入 `run_meta.json`（git commit、参数、库版本、时间戳）。
- `--eval_batch_size` 目前必须为 `1`。

更多命令、图片说明、复现备注和引用请看 [README.md](README.md)。
