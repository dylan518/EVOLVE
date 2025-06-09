#!/usr/bin/env python
"""Run the full EVOLVE pipeline from a single command.

This script expects a TOML configuration file that defines two sections:

[model]   - parameters for starting the vLLM API server
[genome]  - arguments for the Genome / GenomePlus runner (run_genome.py or run_genomeplus.py)
[env]     - optional helper settings (e.g. wait time after starting server)

Example usage:
    python scripts/run_pipeline.py --config config/mmlu_experiment.toml
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Mapping, Any
from contextlib import suppress
import socket
import requests

# tomllib is available from Python 3.11; for 3.10 we fall back to tomli
try:
    import tomllib as tomli  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli  # type: ignore


################################################################################
# Helper functions
################################################################################

def _expand_path(path: str) -> str:
    """Expand ~ and environment variables inside *path*."""
    return os.path.expandvars(os.path.expanduser(path))


def _start_vllm(model_cfg: Mapping[str, Any]) -> subprocess.Popen:
    """Launch the vLLM OpenAI-compatible API server and return the process."""

    # Environment variables required by vLLM
    env = os.environ.copy()
    env.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    env.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    # Extract configuration values with sensible defaults
    model = model_cfg["name"]
    port = model_cfg.get("port", 8000)
    gpu_id = model_cfg.get("gpu_id", 0)
    max_loras = model_cfg.get("max_loras", 20)
    max_lora_rank = model_cfg.get("max_lora_rank", 16)
    gpu_mem_util = model_cfg.get("gpu_memory_utilization", 0.45)
    max_model_len = model_cfg.get("max_model_len", 4096)
    seed = model_cfg.get("seed", 42)

    # Directory for logs
    root_name = model_cfg.get("root") or Path(model).name.replace("/", "-")
    log_dir = Path("vllm_logs") / root_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "port.log", "w")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--trust-remote-code",
        "--enable-lora",
        "--seed",
        str(seed),
        "--max-lora-rank",
        str(max_lora_rank),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--max-loras",
        str(max_loras),
        "--max-cpu-loras",
        str(max_loras),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
    ]

    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[run_pipeline] Launching vLLM server on port {port} …")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    return proc


def _build_genome_command(genome_cfg: Mapping[str, Any]) -> List[str]:
    """Convert the [genome] section into a CLI command list."""

    script = genome_cfg.get("script", "run_genome.py")
    args: List[str] = [sys.executable, script]

    for key, value in genome_cfg.items():
        if key == "script" or value is None:
            continue

        cli_key = f"--{key}"

        if isinstance(value, bool):
            if value:
                args.append(cli_key)
        elif isinstance(value, list):
            args.append(cli_key)
            args.extend(map(str, value))
        else:  # str, int, float
            args.extend([cli_key, str(value)])

    return args


def _is_port_open(port: int) -> bool:
    """Return True if *port* already has a listening TCP server on localhost."""
    with suppress(Exception):
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    return False


def _is_vllm_alive(port: int) -> bool:
    """Return True if an OpenAI-compatible endpoint is responding on *port*."""
    if not _is_port_open(port):
        return False
    with suppress(Exception):
        r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
        return r.status_code == 200
    return False


def _ensure_setup(setup_cfg: Mapping[str, Any]):
    """Run setup script if env/venv not present."""
    env_dir = Path("env")
    if env_dir.is_dir() and (env_dir / "bin" / "activate").exists():
        print("[run_pipeline] Virtual environment detected — skipping setup.")
        return

    script_path = setup_cfg.get("script", "scripts/setup.sh")
    script_path = _expand_path(script_path)
    print(f"[run_pipeline] Running setup script: {script_path}")
    subprocess.run(["bash", script_path], check=True)


################################################################################
# Main entrypoint
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM + Genome pipeline from a single TOML config file.")
    parser.add_argument("--config", required=True, help="Path to the .toml configuration file")
    args = parser.parse_args()

    config_path = Path(_expand_path(args.config)).resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    with config_path.open("rb") as f:
        cfg = tomli.load(f)

    # ------- 0. Optional one-time setup -----------------------------------------
    _ensure_setup(cfg.get("setup", {}))

    # Export Hugging Face token if provided
    hf_token = cfg.get("env", {}).get("huggingface_token")
    if hf_token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        os.environ.setdefault("HF_TOKEN", hf_token)

    # ------- 1. Start vLLM server -------------------------------------------------
    port = cfg["model"].get("port", 8000)
    if _is_vllm_alive(port):
        print(f"[run_pipeline] Detected running vLLM server on port {port} — skipping launch.")
        model_proc = None
    else:
        model_proc = _start_vllm(cfg["model"])

    wait_time = cfg.get("env", {}).get("wait_seconds_after_deploy", 30)
    if model_proc:
        print(f"[run_pipeline] Waiting {wait_time} seconds for the server to warm up …")
        time.sleep(wait_time)

    try:
        # ------- 2. Run Genome / GenomePlus --------------------------------------
        genome_cmd = _build_genome_command(cfg["genome"])
        print(f"[run_pipeline] Executing: {' '.join(genome_cmd)}")
        subprocess.run(genome_cmd, check=True)

    finally:
        # ------- 3. Shutdown -------------------------------------------------------
        if model_proc:
            print("[run_pipeline] Shutting down vLLM server …")
            model_proc.terminate()
            try:
                model_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_proc.kill()
            print("[run_pipeline] Done.")


if __name__ == "__main__":
    main() 