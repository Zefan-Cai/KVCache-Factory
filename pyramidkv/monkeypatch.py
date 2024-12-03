from importlib.metadata import version
import transformers

from pyramidkv.llama_model import llama_flash_attn2_forward_HeadKV, llama_flash_attn2_forward_AdaKV, llama_flash_attn2_forward_PyramidKV,llama_flash_attn2_forward_CAM,llama_flash_attn2_forward_H2O,llama_flash_attn2_forward_SnapKV,llama_flash_attn2_forward_StreamingLLM, llama_flash_attn2_forward_L2Norm
from pyramidkv.llama_model import llama_attn_forward_PyramidKV,llama_attn_forward_CAM,llama_attn_forward_H2O,llama_attn_forward_SnapKV,llama_attn_forward_StreamingLLM, llama_attn_forward_L2Norm
from pyramidkv.llama_model import llama_sdpa_attn_forward_PyramidKV,llama_sdpa_attn_forward_CAM,llama_sdpa_attn_forward_H2O,llama_sdpa_attn_forward_SnapKV,llama_sdpa_attn_forward_StreamingLLM, llama_sdpa_attn_forward_L2Norm
from pyramidkv.llama_model import adaptive_LlamaModel_forward
from pyramidkv.llama_model_think import llama_attn_forward_SnapKV_ThinK, think_model_forward

from pyramidkv.mistral_model import mistral_flash_attn2_forward_AdaKV, mistral_flash_attn2_forward_HeadKV, mistral_flash_attn2_forward_PyramidKV,mistral_flash_attn2_forward_CAM,mistral_flash_attn2_forward_H2O,mistral_flash_attn2_forward_SnapKV,mistral_flash_attn2_forward_StreamingLLM, mistral_flash_attn2_forward_L2Norm
from pyramidkv.mistral_model import mistral_attn_forward_PyramidKV,mistral_attn_forward_CAM,mistral_attn_forward_H2O,mistral_attn_forward_SnapKV,mistral_attn_forward_StreamingLLM, mistral_attn_forward_L2Norm
from pyramidkv.mistral_model import mistral_sdpa_attn_forward_PyramidKV,mistral_sdpa_attn_forward_CAM,mistral_sdpa_attn_forward_H2O,mistral_sdpa_attn_forward_SnapKV,mistral_sdpa_attn_forward_StreamingLLM, mistral_sdpa_attn_forward_L2Norm
from pyramidkv.mistral_model import adaptive_MistralModel_forward

from pyramidkv.llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_generation_llama_new
from pyramidkv.mistral_model import prepare_inputs_for_generation_mistral, prepare_inputs_for_generation_mistral_new


def replace_llama(method, model_name=None):
   
    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV

    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM
        
    elif method == "h2o":
        print("Using H2O!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O
    
    elif method == "cam":
        print("Using CAM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_CAM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_CAM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_CAM
        
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV
    
    elif method == "minference":
        print("Using MInference!")
        from .minference import minference_attn_forward, init_minference
        init_minference(model_name)
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        transformers.models.llama.modeling_llama.LlamaAttention.forward = minference_attn_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minference_attn_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = minference_attn_forward
        
    elif method == "l2norm":
        print("Using L2Norm!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_L2Norm
        
    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_AdaKV

    elif method == "headkv":
        print("Using HeadKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_HeadKV

    elif method == "think":
        print("Using Think!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = think_model_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV_ThinK


    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new

    


def replace_mistral(method):
    
    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_PyramidKV
    
    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_StreamingLLM
        
    elif method == "h2o":
        print("Using H2O!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_H2O

    elif method == "cam":
        print("Using CAM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_CAM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_CAM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_CAM
        
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_SnapKV

    elif method == "l2norm":
        print("Using L2Norm!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_L2Norm
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_L2Norm
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_L2Norm
    
    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_AdaKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "headkv":
        print("Using HeadKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_HeadKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_HeadKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_flash_attn2_forward_HeadKV
    
    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new
