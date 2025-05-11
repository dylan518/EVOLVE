import torch, re
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

TARGET = re.compile(r"(q|k|v|o|gate|up|down)_proj$")   # typical names

def make_lora_state(base_model: str,
                    rank: int = 8,
                    mode: str = "zero",
                    sigma: float = 0.02,
                    dtype = torch.float16):
    """
    Return a dict shaped like `adapter_model.safetensors` but freshly initialised.
    `mode`: "zero"  -> all zeros
            "random"-> N(0, sigma)
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map="cpu"
    )
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        lora_dropout=0.0,
        target_modules=[n for n, _ in model.named_modules() if TARGET.search(n)],
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora = get_peft_model(model, cfg)

    for name, p in lora.named_parameters():
        if "lora_" in name:
            if mode == "zero":
                p.data.zero_()
            else:  # random
                p.data.normal_(0.0, sigma)

    # keep only the adapter tensors
    return {k: v.cpu() for k, v in lora.named_parameters() if "lora_" in k}
