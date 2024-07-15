from typing import List, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel,
	AutoTokenizer,
	AutoModel
)

# Code from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0] #First element of model_output contains all token embeddings
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceEmbedder(object):
	'''
	Receives text and embeds it using sentence transformer model
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

	def embed(
		self,
		texts: List[str],
		batch_size: int = 16,
		convert_to_numpy: bool = True,
		normalize_embeddings: bool = True
	) -> Union[np.array, torch.tensor]:
		embeddings = []
		with torch.no_grad():
			for i in range(0, len(texts), batch_size):
				batch_texts = texts[i:i+batch_size]
				encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
				model_output = self.model(**encoded.to(self.device))

				# Perform pooling
				batch_embeddings = mean_pooling(model_output,encoded['attention_mask'])
				# detach and to cpu
				batch_embeddings = batch_embeddings.detach().cpu()

				# Normalize embeddings
				if normalize_embeddings:
					batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
				embeddings.append(batch_embeddings)

		concatenated_embeddings = torch.cat(embeddings, dim = 0)

		if convert_to_numpy:
			concatenated_embeddings = concatenated_embeddings.numpy()
		# 	concatenated_embeddings = np.concatenate(embeddings, axis = 0)
		# else:
		# 	concatenated_embeddings = torch.cat(embeddings, dim = 0)
		return concatenated_embeddings

if __name__=="__main__":
	from transformers import AutoTokenizer, AutoModel
	
	model_name = "sentence-transformers/all-MiniLM-L6-v2"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name)
	embedder = SentenceEmbedder(
		model = model,
		tokenizer = tokenizer,
		device = torch.device("cpu")
	)
	texts = ["I am hungry", "I am too"]

	embeddings = embedder.embed(texts = texts)
	print(embeddings.shape) # (2, 384)