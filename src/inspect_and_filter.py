# inspect_and_filter.py
import safetensors.torch as satorch
import os
from safetensors.torch import save_file

def process_adapter(input_path, output_dir):
    # 1) Load and inspect
    state = satorch.load_file(input_path)
    allowed = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print("=== PARAM INSPECTION ===")
    
    # 2) Filter and rename weights
    filtered = {}
    for k, v in state.items():
        if any(sub in k for sub in allowed):
            # Remove '.default' from weight names if present
            new_key = k.replace('.default', '')
            filtered[new_key] = v
            print(f"{k:60} => {new_key}")
        else:
            print(f"{k:60} => DROP")
    
    # 3) Save filtered weights
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(filtered, output_path)
    print(f"\nKept {len(filtered)} / {len(state)} tensors")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    input_path = "generated_adapters/ind_000/adapter_model.safetensors"
    output_dir = "generated_adapters/ind_000_filtered"
    process_adapter(input_path, output_dir)