import os
import sys
import traceback
import json
import argparse
from typing import Dict, List

## setup wandb
USE_WANDB = os.environ.get("USE_WANDB", False)
if os.environ["WANDB_ENTITY"]:
	import wandb
	USE_WANDB = True
	from utils.log_utils import wandb_set

import torch

from transformers import (
	PreTrainedTokenizerBase,
	AutoTokenizer,
	AutoModelForCausalLM,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling
)

from utils.data_utils import read_defeasible_inf_records, DefeasibleGenDataset
from utils.hf_argparser import HfArgumentParser

def load_data(fname: str, tokenizer: PreTrainedTokenizerBase, config: dict) -> DefeasibleGenDataset:
	records = read_defeasible_inf_records(
		fname,
		skip_impossible = config.get("skip_impossible", True),
		only_use_weakeners = config.get("only_use_weakeners", True),
	)
	dataset = DefeasibleGenDataset(
		records = records,
		tokenizer = tokenizer,
		mode = config.get("mode", "phu"),
		max_length = config.get("max_length", 256)
	)
	return dataset

def train(config: Dict[str, Dict]):
	## Get Configs
	data_config = config["data"]
	training_config = config["training"]
	logging_config = config["logging"]
	
	## Start run
	run_name = training_config["run_name"]

	## Load Model, Tokenizer
	pretrained_model_name = training_config["pretrained_model"]
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.add_tokens(["[premise]","[hypo]","[strengthener]","[weakener]"])

	if training_config.get("enable_flash_attn2", False):
		model = AutoModelForCausalLM.from_pretrained(
			pretrained_model_name,
			attn_implementation="flash_attention_2"
		)
		print("Applied flash-attn2")
	else:
		model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
	model.resize_token_embeddings(len(tokenizer))
	if training_config.get("enable_bettertransformer", False):
		model = model.to_bettertransformer()
		print("Applied bettertransformer")
	print("Loaded Model, Tokenizer")

	## Load Data
	data_load_config = data_config["load"]
	train_ds = load_data(
		os.path.join(data_config["data_dir"], data_config["train_fname"]),
		tokenizer = tokenizer,
		config = data_load_config
	)
	val_ds = load_data(
		os.path.join(data_config["data_dir"], data_config["val_fname"]),
		tokenizer = tokenizer,
		config = data_load_config
	)
	print("Loaded Train/Val Data", len(train_ds), len(val_ds))

	## Load Trainer
	# prepare args
	weight_save_dir = os.path.join(logging_config["weight_dir"], run_name)
	log_dir = logging_config["log_dir"]

	training_config["output_dir"] = weight_save_dir
	training_config["logging_dir"] = log_dir
	training_args: TrainingArguments = HfArgumentParser(TrainingArguments).parse_dict(training_config, allow_extra_keys = True)[0]


	data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

	#### Init Wandb
	if USE_WANDB:
		wandb_set(run_name, config)
		wandb.watch(model, log="all", log_freq=10)
	else:
		training_config["report_to"].remove("wandb")

	trainer = Trainer(
		model = model,
		args = training_args,
		data_collator = data_collator,
		train_dataset = train_ds,
		eval_dataset = val_ds
	)
	trainer.train()

	## Final Eval
	trainer.evaluate(val_ds, metric_key_prefix = "final")

	## Save Weights
	best_dir = os.path.join(weight_save_dir, "best")
	trainer.save_model(str(best_dir))
	tokenizer.save_pretrained(str(best_dir))
	with open(os.path.join(best_dir, "train_config.json"), 'w') as f:
		json.dump(config, f)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_dir', help = 'config.json file containing data/training arguments')
	args = parser.parse_args()

	with open(args.config_dir, "r") as f:
		config = json.load(f)

	# Run training
	try:
		train(config)
	except Exception as e:
		traceback.print_exc()
		sys.exit(-1)