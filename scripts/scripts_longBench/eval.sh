export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK
max_capacity_prompts=$3 # 128,2048 in paper
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
source_path=$5
model_path=$6
merge_method=$7 # Support "pivot"(LOOK-M_PivotMerge), "weighted"(KVMerger-style weighted merge).
quant_method=$8 # Support kivi and kvquant, default None.
nbits=$9 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir

extra_args=()
if [ -n "${merge_method}" ] && [ "${merge_method}" != "none" ] && [ "${merge_method}" != "None" ]; then
    extra_args+=(--merge "${merge_method}")
fi
if [ -n "${quant_method}" ] && [ "${quant_method}" != "none" ] && [ "${quant_method}" != "None" ]; then
    extra_args+=(--quant_method "${quant_method}")
    if [ -n "${nbits}" ]; then
        extra_args+=(--nbits "${nbits}")
    fi
fi

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    "${extra_args[@]}"
