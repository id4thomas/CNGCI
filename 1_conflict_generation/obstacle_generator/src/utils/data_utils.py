import os
import json
from typing import List, Dict

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

INPUT_TEMPLATE = {
	"phu": "[premise] {p} [hypo] {h} [{u_type}] {u}",
	"hu": "[hypo] {h} [{u_type}] {u}",
}

def read_jsonl_file(fname):
	with open(fname, "r") as f:
		lines = f.readlines()
	data = [json.loads(l.strip()) for l in lines]
	return data

def read_defeasible_inf_records(
		fname: str,
		skip_impossible: bool = True,
		only_use_weakeners: bool =  True
	) -> Dict[str, str]:
	'''
	Read defeasible inference dataset
	Args:

	Returns:
		{
			"premise": "...", "hypothesis": "...", "update": "...", "update_type": "..."
		}
	'''
	if not os.path.isfile(fname):
		raise Exception(f"{fname} is not file")
	
	data = read_jsonl_file(fname)
	record_dict = {
		"premise": [],
		"hypothesis": [],
		"update": [],
		"update_type": [], # strengthener/weakener
	}
	if "phu" in data[0]:
		mode = "phu"
	else:
		mode = "hu"
	for record in tqdm(data):
		# Skip records with no update
		if record["UpdateTypeImpossible"]==True and skip_impossible:
			continue
		hypothesis=record["Hypothesis"]
		update=record["Update"]
		update_type=record["UpdateType"]
		if only_use_weakeners and update_type!="weakener":
			continue

		if mode=="phu":
			premise=record["Premise"]
		else:
			premise = ""
			#premise;hypothesis;update   

		record_dict["premise"].append(premise)
		record_dict["hypothesis"].append(hypothesis)
		record_dict["update"].append(update)
		record_dict["update_type"].append(update_type)
	return record_dict

class DefeasibleGenDataset(Dataset):
	def __init__(
		self,
		records: Dict[str, List[str]],
		tokenizer: PreTrainedTokenizerBase,
		mode: str = "phu",
		max_length: int = 256
	):
		self.records = records
		self.tokenizer = tokenizer
		self.input_template = INPUT_TEMPLATE[mode]
		self.max_length = max_length

		self._num_records = len(records["hypothesis"])
		if mode not in ["phu", "hu"]:
			raise ValueError(f"mode should be one of phu, hu - provided: {mode}")
		self.mode = mode

	def __len__(self) -> int:
		return self._num_records

	def __getitem__(self, idx: int):
		hypothesis = self.records["hypothesis"][idx]
		update = self.records["update"][idx]
		contents = {"h": hypothesis, "u": update}
		if self.mode=="phu":
			contents["p"] = self.records["premise"][idx]

		label = self.records["update_type"][idx]
		contents["u_type"] = label
		input_text = self.input_template.format(**contents)

		## Encode
		encoded = self.tokenizer(
			input_text,
			max_length = self.max_length,
			padding = "max_length",
			truncation = True,
			return_tensors = "pt"
		)
		return encoded