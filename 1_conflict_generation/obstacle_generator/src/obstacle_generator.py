from typing import Tuple, List
import spacy
import torch
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel,
	TextGenerationPipeline
)

from nltk.tokenize import sent_tokenize

class ObstacleGenerator(object):
	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		device: torch.device = torch.device('cuda'),
		mode = "phu"
	):
		model.eval()
		# TextGenerationPipeline
		self.generator = TextGenerationPipeline(
			model = model,
			tokenizer = tokenizer,
			device = device
		)
		self.mode = mode

	def _make_defeasible_gen_input(self, context: str, goal: str):
		return f"<|startoftext|>[premise] {context} [hypo] {goal} [weakener]"
	
	def _generate_text(self, text: List[str], decode_params: dict):
		with torch.no_grad():
			generated = self.generator(
				text,
				return_text = True,
				return_full_text = False,
				**decode_params
			)
		generated_text: List[List[str]] = []
		for text_generated in generated:
			if isinstance(text_generated, list):
				text_generated_text = [x["generated_text"].strip() for x in text_generated]
			else:
				text_generated_text = [text_generated["generated_text"].strip()]
				pass
			generated_text.append(text_generated_text)
		return generated_text
	
	def _get_first_sentence(self, text: str):
		sentences = sent_tokenize(text)
		first_sentence = sentences[0].strip()
		return first_sentence
	
	def _generate_obstacles(self, contexts: List[str], goals: List[str], decode_params: dict) -> List[List[str]]:
		input_texts = [self._make_defeasible_gen_input(c, g) for c,g in zip(contexts, goals)]
		generated_texts = self._generate_text(input_texts, decode_params)

		## only get first sentences
		obstacles = []
		for sample_generated_texts in generated_texts:
			sample_obstacles = list(map(self._get_first_sentence, sample_generated_texts))
			obstacles.append(sample_obstacles)
		return obstacles
		
	def generate(
		self,
		contexts: List[str],
		goals: List[str],
		decode_params: dict
	):
		obstacles = self._generate_obstacles(contexts, goals, decode_params=decode_params)
		return obstacles