import itertools
from typing import Tuple, List
import spacy
import neuralcoref
from nltk.tokenize import sent_tokenize

import torch
import torch.nn as nn
import numpy as np
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel
)

from src.text_generator import TextGenerator
from src.sentence_embedder import SentenceEmbedder
from src.story_dataclasses import CommonsenseRelation

CATEGORIES = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

class CommonsenseRelationSentenceGenerator(TextGenerator):
	'''
	Use comet model to generate relation sentences
	'''
	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		device: torch.device = torch.device('cuda'),
	):
		super().__init__(
			model = model,
			tokenizer = tokenizer,
			device = device,
			generation_mode="text2text"
		)

	def generate(
		self,
		text: str,
		relation_types: List[str],
		decode_params: dict,
		batch_size = 8
	) -> List[List[str]]:
		'''
		generated_texts: [['rel1_sent1', 'rel1_sent2',..], ['rel2_sent1',...]]
		'''
		input_texts = [self._make_input(text, relation_type) for relation_type in relation_types]
		generated_texts: List[List[str]] = self._generate_text(
			input_texts,
			decode_params = decode_params,
			batch_size = batch_size
		)
		return generated_texts

	def _make_input(self, text: str, relation: str):
		return f"{text} {relation} [GEN]"

class CommonsenseRelationGenerator(object):
	'''
	* 1. Uses comet model to generate relation
	* 2. Embeds Relation sentences
	'''
	def __init__(
		self,
		comet_model: PreTrainedModel,
		comet_tokenizer: PreTrainedTokenizerBase,
		embedding_model: nn.Module,
		device: torch.device = torch.device('cuda')
	):
		self.relation_sentence_generator = CommonsenseRelationSentenceGenerator(
			model = comet_model,
			tokenizer = comet_tokenizer,
			device = device
		)

		self.embedder = SentenceEmbedder(model = model, device = device)
		
	def generate(
		self,
		text: str,
		relation_types: List[str],
		decode_params: dict,
		text_generator_batch_size: int = 8,
		text_embedder_batch_size: int = 8
	) -> List[CommonsenseRelation]:
		# list of relation sentences per relation

		relation_sentences: List[List[str]] = self.CommonsenseRelationSentenceGenerator(
			text = text,
			relation_types = relation_types,
			decode_params = decode_params,
			batch_size = text_generator_batch_size
		)

		## Embed
		embedding_input_texts: List[str] = itertools.chain(*relation_sentences)
		embeddings: np.array = self.embedder.embed(
			texts = embedding_input_texts,
			batch_size = text_embedder_batch_size,
			convert_to_numpy = True,
			normalize_embeddings = True
		)

		## Unpack Embeddings
		idx = 0
		relations = []
		for relation_type, type_sentences in zip(relation_types, relation_sentences):
			relation = CommonsenseRelation(
				relation_type = relation_type,
				values = type_sentences,
				embeddings = embeddings[idx:idx+len(type_sentences)]
			)
			relations.append(relation)

		return relations

if __name__=="__main__":
	from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

	model_dir = "" # COMET-2020 model here
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
	generator = CommonsenseRelationSentenceGenerator(
		model = model,
		tokenizer = tokenizer,
		device = torch.device("cpu")
	)

	generated = generator.generate(
		text = "I am hungry",
		relation_types = ["xWant", "xNeed"],
		decode_params = {
			"num_beams": 5,
			"num_return_sequences": 5
		}
	)

	'''
	[
		['to eat something','to eat','to go to the kitchen','to go to the store','to go to the bathroom'],
		['to go to the store','to go to the grocery store','to go to the kitchen','none','to eat something']
	]
	'''
	print(generated)
	