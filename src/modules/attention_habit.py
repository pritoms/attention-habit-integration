import logging
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_transformers import BertConfig, BertModel


logger = logging.getLogger(__name__)


class AttentionHabit(nn.Module):

    def __init__(self, parent_size: int, child_size: int, attention_heads: int, attention_dim: int) -> None:
        super(AttentionHabit, self).__init__()
        self.parent_size = parent_size
        self.child_size = child_size
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim

        self.query_projection = nn.Linear(self.parent_size, self.attention_dim)
        self.key_projection = nn.Linear(self.parent_size, self.attention_dim)
        self.value_projection = nn.Linear(self.parent_size, self.attention_dim)
        self.out_projection = nn.Linear(self.attention_dim, self.parent_size)

    def forward(self, init_data: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        # Embed the init data into the same space as input data.
        init_data = self._embedding(init_data)

        # Compute the query, key, and value tensors.
        query, key, value = self._compute_query_key_value(init_data, input_data)

        # Compute the attention weights and attention output.
        attention_weights = self._compute_attention_weights(query, key)
        attention_output = self._compute_attention_output(attention_weights, value)

        return attention_output

    def _embedding(self, init_data: torch.Tensor) -> torch.Tensor:
        embedding = nn.Linear(self.child_size, self.parent_size)
        init_data = embedding(init_data)

        return init_data

    def _compute_query_key_value(self, init_data: torch.Tensor, input_data: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        query = self.query_projection(input_data)
        key = self.key_projection(init_data)
        value = self.value_projection(init_data)

        return [query, key, value]

    def _compute_attention_weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attention_weights = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = attention_weights / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)

        return attention_weights

    def _compute_attention_output(self, attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_output = torch.matmul(attention_weights, value)
        attention_output = self.out_projection(attention_output)

        return attention_output
