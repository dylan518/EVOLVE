import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

def ensure_template(base_model: str, output_dir: str, rank: int, target_modules: list):
    """
    Ensure the output directory has the LoRA config and tokenizer.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        # Save LoRA config
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 4,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
        lora_cfg.save_pretrained(output_dir)
        # Save tokenizer if available
        try:
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tok.save_pretrained(output_dir)
        except Exception:
            pass

def make_lora_state(
    base_model: str,
    rank: int,
    mode: str,
    sigma: float,
    dtype: torch.dtype,
    target_modules: list
) -> dict:
    """
    Build a LoRA state dict (zero- or random-init) for specified modules.
    Returns weights in the format expected by vLLM (without .default in weight names).
    """
    # Load base model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu"
    )
    # Set up LoRA
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, lora_cfg)
    
    # Initialize weights
    for name, param in lora_model.named_parameters():
        if "lora_" in name:
            if mode == "zero":
                param.data.zero_()
            else:
                param.data.normal_(0.0, sigma)
    
    # Extract LoRA parameters and ensure correct naming format
    state_dict = {}
    for k, v in lora_model.state_dict().items():
        if "lora_" in k:
            # Remove .default from weight names if present
            new_key = k.replace('.default', '')
            state_dict[new_key] = v.cpu()
    
    return state_dict

def main():
    parser = argparse.ArgumentParser(
        description="Generate adjustable LoRA adapters for a base model"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="HuggingFace model ID or local path for the base LLM"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_adapters",
        help="Directory to save adapters"
    )
    parser.add_argument(
        "--num_adapters", type=int, default=1,
        help="Number of adapter checkpoints to generate"
    )
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA rank (r)"
    )
    parser.add_argument(
        "--mode", choices=["zero", "random"], default="zero",
        help="Initialization mode for LoRA parameters"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.02,
        help="Stddev for random init"
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "float32"],
        default="float16",
        help="Data type for model weights"
    )
    parser.add_argument(
        "--target_modules", type=str, default="q_proj,v_proj",
        help="Comma-separated modules to target (e.g., q_proj,v_proj)"
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # Prepare config/tokenizer template
    ensure_template(args.base_model, args.output_dir, args.rank, target_modules)

    for idx in range(args.num_adapters):
        adapter_name = f"ind_{idx:03d}"
        folder = os.path.join(args.output_dir, adapter_name)
        os.makedirs(folder, exist_ok=True)
        # Copy config/tokenizer files
        for fname in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = os.path.join(args.output_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(folder, fname)
                if not os.path.isfile(dst):
                    import shutil; shutil.copy(src, dst)
        # Build and save LoRA state
        state = make_lora_state(
            args.base_model, args.rank, args.mode, args.sigma, dtype, target_modules
        )
        tensor_path = os.path.join(folder, "adapter_model.safetensors")
        save_file(state, tensor_path)
        print(f"Saved LoRA adapter '{adapter_name}' ({len(state)} tensors) to {folder}")

if __name__ == '__main__':
    main()