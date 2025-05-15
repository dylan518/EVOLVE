import os
import random
import uuid
from typing import Dict, List

import torch
from loguru import logger
from src.utils import save_lora_weight
from src.base.base_individual import BaseIndividual

class Individual(BaseIndividual):
    def __init__(
        self, id: str, x: Dict, parent: List[str], 
        weight_path: str, model_name_or_path: str, lora_config_path: str,
        seed: int = 42, from_mutation: str = None, lora_config=None
    ):
        super().__init__(
            id=id, x=x, weight_path=weight_path,
            model_name_or_path=model_name_or_path,
            lora_config_path=lora_config_path, seed=seed
        )
        self.parent = parent
        self.from_mutation = from_mutation
        self.lora_config = lora_config
        
    def save_individual(self, save_path):
        save_lora_weight(
            lora_weight=self.x,
            lora_path=save_path,
            tokenizer=self.tokenizer,
            config=self.lora_config  # Use the stored LoraConfig object
        )
        self.weight_path = save_path

    def mutation(self, individual_mutation_rate: float, gene_mutation_rate: float, sigma: float=0.01, best_fitness_score: float=-100):
        # if self.fitness_score >= best_fitness_score:
        #     logger.info(f"Individual {self.id} (fitness score: {self.fitness_score:.4f}) is the best individual, no mutation")
        #     # do nothing
        #     return
        if random.random() > individual_mutation_rate:
            # donot mutate, do nothing
            return None
        else:
            mutated_weights = {}
            for key, tensor in self.x.items():
                device = tensor.device
                dtype = tensor.dtype

                mutation_mask = torch.rand(tensor.shape, device=device) < gene_mutation_rate
                noise = torch.randn(tensor.shape, device=device, dtype=dtype) * sigma

                tensor_mutated = tensor + noise * mutation_mask.to(dtype)
                mutated_weights[key] = tensor_mutated
            
            # reset individual state
            old_id = self.id
            new_id = uuid.uuid4().hex

            logger.info(f"Individual {old_id} (fitness score: {self.fitness_score:.4f}) mutated into {new_id}")
            
            new_weight_path = os.path.join(os.path.dirname(self.weight_path), f"ind_{new_id}")
            os.makedirs(new_weight_path, exist_ok=True)
            
            # Create new individual with the mutated weights
            new_individual = Individual(
                id=new_id, 
                x=mutated_weights, 
                parent=[self.id],
                weight_path=new_weight_path, 
                model_name_or_path=self.model_name_or_path,
                lora_config_path=self.config_path, 
                from_mutation=old_id,
                lora_config=self.lora_config  # Pass along the LoraConfig
            )
            
            # Save the individual's weights
            new_individual.save_individual(new_weight_path)
            
            return new_individual