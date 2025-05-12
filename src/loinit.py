import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

TARGET = re.compile(r"(q|k|v|o|gate|up|down)_proj$")
TEMPLATE_DIR = "generated_adapters"          # central scratch folder


# ------------------------------------------------------------------
# helper: make sure TEMPLATE_DIR has adapter_config.json (+tokenizer)
# ------------------------------------------------------------------
def _ensure_template(base_model: str, rank: int = 8):
    cfg_file = os.path.join(TEMPLATE_DIR, "adapter_config.json")
    if os.path.isfile(cfg_file):            # already good
        return
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    # 1) minimal LoRA config
    LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    ).save_pretrained(TEMPLATE_DIR)

    # 2) tokenizer (helpful for later reloads; ignore download errors)
    try:
        AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)\
                     .save_pretrained(TEMPLATE_DIR)
    except Exception:
        pass


# ------------------------------------------------------------------
# public API: build fresh LoRA tensors
# ------------------------------------------------------------------
def make_lora_state(
    base_model: str,
    rank: int = 8,
    mode: str = "zero",          # "zero" | "random"
    sigma: float = 0.02,
    dtype = torch.float16,
) -> dict:
    """
    Return a dict shaped like adapter_model.safetensors, and guarantee
    that TEMPLATE_DIR now contains adapter_config.json.
    """
    _ensure_template(base_model, rank)      # ← writes the config once

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map="cpu"
    )

    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        lora_dropout=0.0,
        target_modules=[n for n, _ in model.named_modules()
                        if TARGET.search(n)],
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, cfg)

    for name, p in lora_model.named_parameters():
        if "lora_" in name:
            if mode == "zero":
                p.data.zero_()
            else:                # "random"
                p.data.normal_(0.0, sigma)

    return {k: v.cpu() for k, v in lora_model.named_parameters()
            if "lora_" in k}
