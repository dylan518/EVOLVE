from typing import List
import os
import glob

import time
from peft import LoraConfig
from transformers import AutoTokenizer
from loguru import logger
from safetensors.torch import load_file, save_file

def get_base_url(ports: List[int]) -> List[str]:
    """Get base urls from ports."""
    return [f"http://0.0.0.0:{port}/v1" for port in ports]

def get_lora_pools(lora_dir: str) -> List[str]:
    """do not change this function, it is used for Model Swarms / Genome / GenomePlus."""
    lora_pools = []
    for dirname in [
        "code_alpaca", "gpt4_alpaca", "cot", "lima", "oasst1", "open_orca", "flan_v2", "science_literature", "wizardlm", "sharegpt",
    ]:
        path = os.path.join(lora_dir, dirname)
        if os.path.exists(path):
            lora_pools.append(path)
        
    return lora_pools

def load_lora_weight(lora_path: str) -> dict:
    """Load LoRA weights from path."""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weight not found at {lora_path}")
    return load_file(os.path.join(lora_path, "adapter_model.safetensors"))

def save_lora_weight(lora_weight, lora_path: str, tokenizer: AutoTokenizer | str, config: LoraConfig | str = None):
    assert tokenizer is not None, "Tokenizer must be provided (for vllm evaluate)."
    
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # If config is None, try to load from lora_path
    if config is None:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            config = LoraConfig.from_pretrained(lora_path)
        else:
            # Create a default config if none exists
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
    elif isinstance(config, str):
        config = LoraConfig.from_pretrained(config)
    
    # Ensure the directory exists
    os.makedirs(lora_path, exist_ok=True)
    
    # Save tokenizer and config
    tokenizer.save_pretrained(lora_path)
    config.save_pretrained(lora_path)
    
    # Save the weights
    save_file(lora_weight, filename=os.path.join(lora_path, "adapter_model.safetensors"))
    
    # Wait for save to complete
    time.sleep(1)
    
def get_gemma_prompt(user_question):
    template = f"""
    <start_of_turn>user
    {user_question}
    <end_of_turn>
    <start_of_turn>model
    """
    return template

def get_llama3_1_prompt(user_question):
    template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{user_question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""
    return template