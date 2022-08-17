import logging
import os
import torch
from torch import nn
from pytorch_transformers import BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from src.modules.attention_habit import AttentionHabit


logger = logging.getLogger(__name__)


class BertAttentionHabit(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAttentionHabit, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.attention_habit = AttentionHabit(config.hidden_size, config.hidden_size, config.num_attention_heads, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                init_data=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        attention_output = self.attention_habit(init_data, pooled_output)

        logits = self.classifier(attention_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
