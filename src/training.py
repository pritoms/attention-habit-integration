import logging
import os
import random
import time
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from src.data_processing import TextDataProcessing
from src.models import BertAttentionHabit


logger = logging.getLogger(__name__)


class Training:
    def __init__(self, path_to_pretrained_model: str, path_to_trainingset: str, path_to_validationset: str, path_to_testset: str, batch_size: int, epochs: int) -> None:
        self._path_to_pretrained_model = path_to_pretrained_model
        self._path_to_trainingset = path_to_trainingset
        self._path_to_validationset = path_to_validationset
        self._path_to_testset = path_to_testset
        self._batch_size = batch_size
        self._epochs = epochs

        self._trainingset = None
        self._validationset = None
        self._testset = None

    @property
    def trainingset(self):
        return self._trainingset

    @property
    def validationset(self):
        return self._validationset

    @property
    def testset(self):
        return self._testset

    def read_csv(self) -> None:
        if self._trainingset is None:
            self._trainingset = TextDataProcessing.read_csv(self._path_to_trainingset)

        if self._validationset is None:
            self._validationset = TextDataProcessing.read_csv(self._path_to_validationset)

        if self._testset is None:
            self._testset = TextDataProcessing.read_csv(self._path_to_testset)

    def balance_classes(self) -> None:
        self._trainingset = TextDataProcessing.balance_classes(self._trainingset)

    def split_train_validation(self) -> None:
        self._trainingset, self._validationset = TextDataProcessing.split_train_validation(self._trainingset)

    def preprocess(self, max_length: int) -> None:
        self._trainingset.loc[:, 'input_ids'], self._trainingset.loc[:, 'attention_mask'] = TextDataProcessing.preprocess(self._trainingset, self._tokenizer, max_length=max_length)
        self._validationset.loc[:, 'input_ids'], self._validationset.loc[:, 'attention_mask'] = TextDataProcessing.preprocess(self._validationset, self._tokenizer, max_length=max_length)
        self._testset.loc[:, 'input_ids'], self._testset.loc[:, 'attention_mask'] = TextDataProcessing.preprocess(self._testset, self._tokenizer, max_length=max_length)

    def dataloader(self) -> (DataLoader, DataLoader, DataLoader, DataLoader):
        train_dataset = TensorDataset(self._trainingset['input_ids'].to_numpy(), self._trainingset['attention_mask'].to_numpy(), self._trainingset['label'].to_numpy())
        valid_dataset = TensorDataset(self._validationset['input_ids'].to_numpy(), self._validationset['attention_mask'].to_numpy(), self._validationset['label'].to_numpy())
        test_dataset = TensorDataset(self._testset['input_ids'].to_numpy(), self._testset['attention_mask'].to_numpy())

        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False)
        return [train_dataloader, valid_dataloader, test_dataloader]

    def train(self) -> Tuple[torch.Tensor]:
        model = BertAttentionHabit.from_pretrained(self._path_to_pretrained_model, num_labels=1, output_attentions=False, output_hidden_states=False)
        model.cuda()

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        epochs = self._epochs
        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a backward pass.
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in valid_dataloader:

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete!")

        return model

    def predict(self, model: torch.Tensor, testloader: DataLoader, max_length: int) -> None:
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in testloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)

        print('DONE.')
