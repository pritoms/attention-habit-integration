import logging
import os
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

class TextDataProcessing:
    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data

    @staticmethod
    def balance_classes(data: pd.DataFrame) -> pd.DataFrame:
        data_pos = data[data['label'] == 1]
        data_neg = data[data['label'] == 0]

        # Drop the extra negative reviews from the training set.
        drop_indices = np.random.choice(data_neg.index, len(data_pos) - len(data_neg), replace=False)
        data_neg = data_neg.drop(drop_indices)

        balanced_data = pd.concat([data_pos, data_neg], axis=0)
        return balanced_data

    @staticmethod
    def split_train_validation(data: pd.DataFrame) -> List[pd.DataFrame]:
        train, validation = train_test_split(data, test_size=0.33, random_state=42)
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        return [train, validation]

    @staticmethod
    def preprocess(data: pd.DataFrame, tokenizer: object, max_length: int) -> (List[int], List[int]):
        input_ids = []
        attention_mask = []

        for text in data['text']:
            encoded_dict = tokenizer.encode_plus(text=text, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        return [input_ids, attention_mask]

    @staticmethod
    def tokenize(text: str, tokenizer: object) -> (List[int], List[int]):
        encoded_dict = tokenizer.encode_plus(text=text, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        return [input_ids, attention_mask]
