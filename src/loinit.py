import os, torch, re, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

TARGET = re.compile(r"(q|k|v|o|gate|up|down)_proj$")   # typical proj layers

def make_lora_state(base_model: str, rank: int = 8,
                    mode: str = "zero", sigma: float = 0.02,
                    dtype = torch.float16):
    """
    Return a state-dict shaped like adapter_model.safetensors.
    mode = zero → all-zeros;  random → N(0, σ).
    """
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=dtype,
                                                 device_map="cpu")
    cfg = LoraConfig(
        r=rank, lora_alpha=rank * 4, lora_dropout=0.0,
        target_modules=[n for n, _ in model.named_modules()
                        if TARGET.search(n)],
        bias="none", task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, cfg)

    for name, p in lora_model.named_parameters():
        if "lora_" in name:
            if mode == "zero":
                p.data.zero_()
            else:
                p.data.normal_(0.0, sigma)

    return {k: v.cpu() for k, v in lora_model.named_parameters()
            if "lora_" in k}
