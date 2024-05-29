import torch
from transformers import BitsAndBytesConfig , CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, AutoTokenizer , LlamaTokenizer , AutoModelForCausalLM
# from loader.llamatokenizer import Tokenizer 

quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def load_vision_model(pretrained_model, device = "cpu", cache_dir = None ):
    vision_model = CLIPVisionModel.from_pretrained(pretrained_model, cache_dir=cache_dir)
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model,  cache_dir=cache_dir)
    return vision_model , image_processor;

def load_llm(pretrained_model, device = "cpu", cache_dir = None , quantization_config = None , attn_implementation = None):
    if "SweatyCrayfish/llama-3-8b-qua" in pretrained_model:
        llm = AutoModelForCausalLM.from_pretrained("SweatyCrayfish/llama-3-8b-quantized", device_map="auto", load_in_4bit=True)
    elif attn_implementation is None:
        llm = AutoModelForCausalLM.from_pretrained(pretrained_model,  cache_dir=cache_dir, quantization_config = quantization_config).to(device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(pretrained_model,  cache_dir=cache_dir, quantization_config = quantization_config, attn_implementation = attn_implementation)
    
    # if "meta-llama/" in pretrained_model:
    #     tokenizer = tokenizer = Tokenizer("llama_tokenizer.model")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir )
    return llm, tokenizer

# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

