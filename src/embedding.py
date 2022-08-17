import logging
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class Embedding:
    def __init__(self, path_to_pretrained_model: str) -> None:
        self._tokenizer = BertTokenizer.from_pretrained(path_to_pretrained_model)

    @property
    def tokenizer(self):
        return self._tokenizer

    def batch_embedding(self, data: pd.DataFrame, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = TextDataProcessing.preprocess(data, self._tokenizer, max_length=max_length)

        return [input_ids, attention_mask]

    def embedding(self, text: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = TextDataProcessing.tokenize(text, self._tokenizer)

        return [input_ids, attention_mask]
