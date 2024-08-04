from typing import Tuple, List
import spacy
import neuralcoref
from nltk.tokenize import sent_tokenize

import torch
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel
)

from src.modules.text_generator import TextGenerator
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
		story: ConflictStory,
		decode_params: dict
	) -> List[str]:
		input_text = self._make_candidate_gen_input(story = story)
		for i in range(5):
			try:
				generated_texts = self._generate_text([input_text], decode_params)[0]
				candidates = list(map(self._get_first_sentence, generated_texts))
			except Exception as e:
				if i!=4:
					continue
				else:
					raise e
		return candidates

	def _make_candidate_gen_input(self, story: ConflictStory) -> str:
		raise NotImplementedError("This method needs to be implemented")

	def _get_first_sentence(self, text: str, min_length: int = 5):
		## Cases with Special Tokens
		if "<obs>" in text:
			text = text.split("<obs>")[0]
		if "<story>" in text:
			splits = text.split("<story>")
			if len(splits[0].strip())<min_length:
				text = splits[1].strip()
			else:
				text = splits[0].strip()
		if "<ctx>" in text:
			splits = text.split("<ctx>")
			if len(splits[0].strip())<min_length:
				text = splits[1].strip()
			else:
				text = splits[0].strip()

		sentences = sent_tokenize(text)
		if len(sentences)==0:
			return ""
		first_sentence = sentences[0].strip()
		if len(first_sentence)<min_length:
			return ""
		
		if first_sentence[-1] not in [".", "!", "?"]:
			first_sentence = first_sentence+"."
		return first_sentence	

class LMNextSentenceCandidateGenerator(NextSentenceCandidateGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def _make_candidate_gen_input(self, story: ConflictStory) -> str:
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

	def _make_candidate_gen_input(self, story: ConflictStory) -> str:
		'''
		Sentence Order: Context, Obstacle, S2, S4, S5 (if S3 is Obstacle)
		template: bos_token + context+ " " + obs + " " + preceding_generations
		'''
		# prefix_template = '{bos}{context} {obstacle}'
		prefix_template = '{bos} initial: {context} obstacle: {obstacle} story: '
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
	
class ObsLM245NextSentenceCandidateGeneratorV2(NextSentenceCandidateGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _make_candidate_gen_input(self, story: ConflictStory) -> str:
		'''
		Sentence Order: `<ctx> ... <obs> ... <story><ctx> S2 <obs> S4 S5`
		'''
		# prefix_template = '{bos}{context} {obstacle}'
		prefix_template = '<ctx> {context} <obs> {obstacle} <story><ctx>'
		prefix = prefix_template.format(
			bos = self.tokenizer.bos_token,
			context = story.sentences[story.context_idx].value,
			obstacle = story.sentences[story.obstacle_idx].value,
		)
		
		if story.num_sentences>2: ## generating s4~
			preceding_generations = []
			## Add S2 & Obs
			
			for i in range(story.num_sentences):
				if i==story.context_idx:
					continue
				elif i==story.obstacle_idx:
					preceding_generations.append("<obs>")
				preceding_generations.append(story.sentences[i].value)
			preceding_generations = " ".join(preceding_generations)
		else:
			preceding_generations = ""
		
		input_text = " ".join([prefix, preceding_generations])
		# print("INPUT TEXT:", repr(input_text))
		input_text = " ".join([prefix, preceding_generations]).strip()
		return input_text

if __name__=="__main__":
	# TODO - Implement candidate gen example
	pass