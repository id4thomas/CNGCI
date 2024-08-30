import argparse
import json
import logging
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from data_types import KBData, ROCData, ModelConfig
from torch_datasets.kb_datasets import KBDataset
from torch_datasets.roc_datasets import ROCDataset
from utils.hf_argparser import HfArgumentParser


logger = logging.getLogger(__name__)

USE_WANDB = False
if os.environ.get("WANDB_ENTITY", ""):
    import wandb

    USE_WANDB = True


def set_random_seed(seed: int = 100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main(config: Dict[str, Any]):
    # 1. Parse Args
    data_config: dict = config["data"]
    data_mode: str = data_config["mode"]
    if data_mode == "roc":
        train_data = ROCData(**data_config["train"])
        val_data = ROCData(**data_config["val"])
        dataset_class = ROCDataset
    elif data_mode == "kb":
        train_data = KBData(**data_config["train"])
        val_data = KBData(**data_config["val"])
        dataset_class = KBDataset
    else:
        raise ValueError("data mode {} not allowed".format(data_mode))

    model_config = ModelConfig(**config["model"])
    trainer_config: dict = config["trainer"]
    result_config: dict = config["result"]

    # Load Tokenizer, Prepare Datasets
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model)
    if model_config.added_special_tokens:
        tokenizer.add_special_tokens(model_config.added_special_tokens)
    if model_config.added_tokens:
        tokenizer.add_tokens(model_config.added_tokens)
    train_ds = dataset_class(train_data, tokenizer)
    val_ds = dataset_class(val_data, tokenizer)

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(model_config.pretrained_model)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare Weight Directory
    weight_dir: str = result_config["weight_dir"]
    trainer_config["output_dir"] = weight_dir

    # Prepare Trainer
    effective_batch_size = (
        trainer_config.get("per_device_train_batch_size", 1)
        * trainer_config.get("gradient_accumulation_steps", 1)
        * torch.cuda.device_count()
    )
    logger.info("Effective Batch Size: {}".format(effective_batch_size))

    if not USE_WANDB:
        report_to: List[str] = trainer_config.get("report_to", list())
        if "wandb" in report_to:
            report_to.remove("wandb")
        trainer_config["report_to"] = report_to

    trainer_args = HfArgumentParser(TrainingArguments).parse_dict(
        trainer_config, allow_extra_keys=True
    )[0]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # Final Evaluation
    trainer.evaluate(val_ds, metric_key_prefix="final")

    # Save Models
    best_dir = os.path.join(weight_dir, "best")
    os.makedirs(best_dir)

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    with open(os.path.join(best_dir, "run_config.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="config.json",
        help="config.json file containing data/training arguments",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config)
