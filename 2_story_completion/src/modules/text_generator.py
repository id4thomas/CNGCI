from typing import List
import torch 
from transformers import (
	PreTrainedTokenizerBase,
	PreTrainedModel,
	Text2TextGenerationPipeline, ## enc-dec
	TextGenerationPipeline ## dec
)

PIPELINE_TYPE = {
	"text": TextGenerationPipeline,
	"text2text": Text2TextGenerationPipeline
}

class TextGenerator(object):
	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerBase,
		device: torch.device = torch.device('cuda'),
		generation_mode: str = "text"
	):
		model.eval()
		self.generator = PIPELINE_TYPE[generation_mode](
			model = model,
			tokenizer = tokenizer,
			device = device
		)
		self.tokenizer = tokenizer
		self.generation_mode = generation_mode

	def _generate_text(self, text: List[str], decode_params: dict, batch_size: int = 1) -> List[List[str]]:
		if self.generation_mode=="text":
			decode_params["return_full_text"] = False
		if self.tokenizer.eos_token_id:
			decode_params["eos_token_id"] = self.tokenizer.eos_token_id

		with torch.no_grad():
			generated = self.generator(text, batch_size = batch_size, **decode_params)
		generated_text: List[List[str]] = []
		for text_generated in generated:
			if isinstance(text_generated, list):
				text_generated_text = [x["generated_text"].strip() for x in text_generated]
			else:
				text_generated_text = [text_generated["generated_text"].strip()]
				pass
			generated_text.append(text_generated_text)
		return generated_text