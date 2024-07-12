from typing import List
from peft import (
	get_peft_model,
	LoraConfig,
	TaskType
)
from transformers import PreTrainedModel

'''
GPT2: c_attn
* https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/models/gpt2/modeling_gpt2.py#L623
'''
def get_lora_config(
		lora_r: int = 8,
		lora_alpha: int = 16,
		lora_dropout: float = 0.1,
		target_modules: List[str] = ["c_attn"],
	):
	return LoraConfig(
		task_type=TaskType.CAUSAL_LM, 
		inference_mode=False, 
		r = lora_r, 
		lora_alpha = lora_alpha, 
		lora_dropout = lora_dropout,
		# FOR POLYGLOT
		bias="none",
		target_modules=target_modules
	)

def load_peft_model(model: PreTrainedModel, config: LoraConfig):
	return get_peft_model(model, config)