from typing import List, Dict, Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


from data_types import ROCData


class ROCDataset(Dataset):

    def __init__(self, data: ROCData, tokenizer: PreTrainedTokenizerBase):
        print(data)
        df = pd.read_csv(data.file, sep="\t")
        self.sources = df.source.values.tolist()
        self.tokenizer = tokenizer
        self.tokenization_config = data.tokenization_config

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int):
        eos_token = self.tokenizer.eos_token
        encoded = self.tokenizer(
            self.sources[idx] + eos_token,
            return_tensors="pt",
            **self.tokenization_config
        )
        return encoded
