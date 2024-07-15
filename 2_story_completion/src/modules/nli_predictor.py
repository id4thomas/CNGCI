from typing import List, Union

import torch
import torch.nn as nn

from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel,
	AutoTokenizer,
	AutoModelForSequenceClassification
)

class NLIPredictor(object):
	'''
	Receives text pair and predicts NLI label
	Uses sentence_transformer CrossEncoder
	'''

	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		device: torch.device = torch.device("cuda")
	):
		self.tokenizer = tokenizer
		self.model = model
		self.model.eval()
		self.model.to(device)

		self.device = device

		self.label_mapping = ['contradiction', 'entailment', 'neutral']

	def predict(
		self,
		texts: List[List[str]],
		batch_size: int = 16
	) -> List[str]:
		'''
		given pairs of text -> predict for each pair
		'''
		labels = []
		with torch.no_grad():
			for i in range(0, len(texts), batch_size):
				batch_texts = texts[i:i+batch_size]
				encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
				batch_prediction = self.model(**encoded).logits
				batch_labels = [self.label_mapping[score_max] for score_max in batch_prediction.argmax(dim=1)]
				labels.extend(batch_labels)
		return labels

if __name__=="__main__":
	from transformers import AutoTokenizer, AutoModelForSequenceClassification

	model_name = 'cross-encoder/nli-distilroberta-base'
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name)

	predictor = NLIPredictor(
		model = model,
		tokenizer = tokenizer,
		device = torch.device("cpu")
	)

	texts = [
		["I am hungry", "I eat food"],
		["I am hungry", "I starve"],
	]
	predicted = predictor.predict(texts = texts)

	# ['entailment', 'contradiction']
	print(predicted)

