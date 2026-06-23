# KVCache-Factory

KVCache-Factory 是一个面向长上下文 LLM 推理的 KV cache 方法集合，覆盖压缩、检索、合并和量化等方向。项目从 PyramidKV 扩展而来，目前把多种 KV cache baseline 放到统一的评测入口下。

## 当前支持

- 压缩/检索：`PyramidKV`、`SnapKV`、`H2O`、`StreamingLLM`、`CAM`、`L2Norm`、`AdaKV`、`HeadKV`、`ThinK`
- 量化：`--quant_method kivi`、`--quant_method kvquant`
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
  --max_capacity_prompts 512 \
  --attn_implementation flash_attention_2 \
  --save_dir ./results_long_bench \
  --use_cache True
```

更多命令、图片说明、复现备注和引用请看 [README.md](README.md)。
