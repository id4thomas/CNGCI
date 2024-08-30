from typing import List, Dict, Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


from data_types import KBData


class KBDataset(Dataset):

    def __init__(self, data: KBData, tokenizer: PreTrainedTokenizerBase):
        atomic_df = pd.read_csv(data.atomic_file, sep="\t")
        conceptnet_df = pd.read_csv(data.conceptnet_file, sep="\t")
        self.atomic_sources = atomic_df.source.values.tolist()
        self.conceptnet_sources = conceptnet_df.source.values.tolist()
        self.sources = self.atomic_sources + self.conceptnet_sources
        self.tokenizer = tokenizer
        self.tokenization_config = data.tokenization_config

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.sources[idx], return_tensors="pt", **self.tokenization_config
        )
        return encoded
