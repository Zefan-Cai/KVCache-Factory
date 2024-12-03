export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK
max_capacity_prompts=$3 # 128,2048 in paper
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
source_path=$5
model_path=$6
merge_method=$7 # Support "pivot"(LOOK-M_PivotMerge).
quant_method=$7 # Support kivi and kvquant, default None.
nbits=$8 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --merge ${merge_method} \
    --nbits ${nbits} \
    --quant_method ${quant_method}
