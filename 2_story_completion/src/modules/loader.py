import torch
from transformers import (
	AutoTokenizer,
	AutoModel,
	AutoModelForSequenceClassification,
	AutoModelForSeq2SeqLM
)

from src.modules.commonsense_relation_generator import CommonsenseRelationGenerator
from src.modules.nli_predictor import NLIPredictor
from src.modules.subject_extractor import SubjectExtractor

def load_subject_extractor(model: str = "en_core_web_sm") -> SubjectExtractor:
	extractor = SubjectExtractor(model = model)
	return extractor

def load_commonsense_generator(
	comet_model_dir: str,
	embedding_model_dir: str,
	device = torch.device("cuda")
) -> CommonsenseRelationGenerator:
	## Load Comet Model
	comet_model = AutoModelForSeq2SeqLM.from_pretrained(comet_model_dir)
	comet_tokenizer = AutoTokenizer.from_pretrained(comet_model_dir)

	## Load Embedder Model
	embedding_model = AutoModel.from_pretrained(embedding_model_dir)
	embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_dir)

	generator = CommonsenseRelationGenerator(
		comet_model = comet_model,
		comet_tokenizer = comet_tokenizer,
		embedding_model = embedding_model,
		embedding_tokenizer = embedding_tokenizer,
		device = device
	)
	return generator

def load_nli_predictor(model_dir: str, device = torch.device("cuda")) -> NLIPredictor:
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	predictor = NLIPredictor(
		model = model,
		tokenizer = tokenizer,
		device = device
	)
	return predictor