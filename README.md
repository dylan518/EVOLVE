EVOLVE applies genetic algorithms to optimize LoRA adapters for improved task performance on standard benchmarks. Building on the GENOME+ framework, which evolves pre-trained adapters across datasets, EVOLVE instead starts from scratch—with adapters initialized to zero.

This approach has not been previously explored, as training from scratch typically requires vast amounts of data and compute. Our work proposes novel utility functions to overcome this challenge by maximizing both generality and efficiency from only a small set of example problems.

We are also extending this evolutionary framework to test-time compute-constrained (TTC) models such as Qwen 1.5B R1.

The broader goal of EVOLVE is to demonstrate that backpropagation may not be the only—or optimal—path forward in neural network training. We see genetic algorithms as a promising alternative for training more efficient, general, and adaptive models.


A lot of the code is not in a format that can be run easily right now, but I'm in the processing of containerising the code.

## Quick-start: One-command pipeline

```bash
# Clone & enter the repo
 git clone <your-fork-or-upstream-url>
 cd EVOLVE

# (Optional) copy the sample config and tweak
 cp config/mmlu_experiment.toml my_experiment.toml
 $EDITOR my_experiment.toml     # change ports, tasks, HF token, etc.

# Run everything – environment setup → model server → Genome search:
 python scripts/run_pipeline.py --config my_experiment.toml
```

What the command does:

1. **Environment setup** – If the `env/` virtual-env doesn't exist, it automatically runs `scripts/setup.sh`, creating the environment and installing requirements (CUDA 11.8 wheels by default).
2. **Model deployment** – Launches a vLLM OpenAI-compatible server with the model and LoRA settings from `[model]` in the TOML (or re-uses an already running server on the same port).
3. **Genome / GenomePlus search** – Executes either `run_genome.py` or `run_genomeplus.py` with every CLI flag generated from the `[genome]` section.
4. **Shutdown** – When the experiment finishes, the script terminates the vLLM server it spawned.

All configuration happens in a single TOML file. The included `config/mmlu_experiment.toml` is a minimal example that replicates the quick MMLU run shown in the docs.

### Hugging Face credentials
Add

```toml
[env]
huggingface_token = "YOUR_TOKEN_HERE"
```

to give the script access to gated weights (exported as `HUGGINGFACEHUB_API_TOKEN`).

---

For advanced scenarios (multiple tasks, GenomePlus, larger populations), simply tweak the TOML fields—no code changes required.