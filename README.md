EVOLVE applies genetic algorithms to optimize LoRA adapters for improved task performance on standard benchmarks. Building on the GENOME+ framework, which evolves pre-trained adapters across datasets, EVOLVE instead starts from scratch—with adapters initialized to zero.

This approach has not been previously explored, as training from scratch typically requires vast amounts of data and compute. Our work proposes novel utility functions to overcome this challenge by maximizing both generality and efficiency from only a small set of example problems.

We are also extending this evolutionary framework to test-time compute-constrained (TTC) models such as Qwen 1.5B R1.

The broader goal of EVOLVE is to demonstrate that backpropagation may not be the only—or optimal—path forward in neural network training. We see genetic algorithms as a promising alternative for training more efficient, general, and adaptive models.


A lot of the code is not in a format that can be run easily right now, but I'm in the processing of containerising the code.