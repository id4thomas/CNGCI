from typing import Tuple, List
import spacy
import neuralcoref
from nltk.tokenize import sent_tokenize

import torch
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel
)

from src.text_generator import TextGenerator
from src.story_dataclasses import StorySentence, ConflictStory

class NextSentenceCandidateGenerator(TextGenerator):
	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		device: torch.device = torch.device('cuda'),
	):
		## initialize generator
		super().__init__(
			model = model,
			tokenizer = tokenizer,
			device = device,
			generation_mode="text"
		)

	def generate(
		self,
		contexts: List[str],
		obstacles: List[str],
		previous_sentences: List[List[str]],
		decode_params: dict
	):
		candidates = self._generate_candidates(contexts, obstacles, previous_sentences, decode_params=decode_params)
		return candidates

	def _make_candidate_gen_input(self,):
		raise NotImplementedError("This method needs to be implemented")

	def _generate_candidates(
		self, 
		contexts: List[str], 
		obstacles: List[str],
		previous_sentences: List[List[str]],
		decode_params: dict
	) -> List[List[str]]:
		input_texts = [self._make_defeasible_gen_input(c, o, p) for c,o,p in zip(contexts, goals)]
		generated_texts = self._generate_text(input_texts, decode_params)

		## only get first sentences
		candidates = []
		for sample_generated_texts in generated_texts:
			sample_obstacles = list(map(self._get_first_sentence, sample_generated_texts))
			obstacles.append(sample_obstacles)
		return obstacles

	def _get_first_sentence(self, text: str):
		sentences = sent_tokenize(text)
		first_sentence = sentences[0].strip()
		return first_sentence	

class LMNextSentenceCandidateGenerator(NextSentenceCandidateGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def _make_candidate_gen_input(self, story: ConflictStory):
		'''
		Sentence Order: Context, ..., Obstacle, ...
		template: bos_token + context+ " " + ... + " " + obstacle + " " + ...
		'''
		sentence_values = [story.sentences[i].value for i in range(story.num_sentences)]
		input_text = "{bos}{preceding_story}".format(
			bos = self.tokenizer.bos_token,
			preceding_story = " ".join(sentence_values)
		)
		return input_text

class ObsLM245NextSentenceCandidateGenerator(NextSentenceCandidateGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _make_candidate_gen_input(self, story: ConflictStory):
		'''
		Sentence Order: Context, Obstacle, S2, S4, S5 (if S3 is Obstacle)
		template: bos_token + context+ " " + obs + " " + preceding_generations
		'''
		prefix_template = '{bos}{context} {obstacle}'
		prefix = prefix_template.format(
			bos = self.tokenizer.bos_token,
			context = story.sentences[story.context_idx].value,
			obstacle = story.sentences[story.obstacle_idx].value,
		)
		
		if story.num_sentences>2: ## generating s2
			preceding_generations = []
			for i in range(story.num_sentences):
				if i==story.context_idx or i==story.obstacle_idx:
					continue
				preceding_generations.append(story.sentences[i].value)
			preceding_generations = " ".join(preceding_generations)
		else:
			preceding_generations = ""
		
		input_text = " ".join([prefix, preceding_generations])
		return input_text

if __name__=="__main__":
	# TODO - Implement candidate gen example
	pass