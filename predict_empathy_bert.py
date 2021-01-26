#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, BertForSequenceClassification
import json
import time
import io
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertAdam
from sklearn import metrics

from tqdm.auto import tqdm, trange

data_dir = "data/"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

filename1 = data_dir+"helpful_comments.json" 
filename2 = data_dir+"unhelpful_comments.json"
    
with open(filename1, "r") as read_file:
    helpful_comments = json.load(read_file)
with open(filename2, "r") as read_file:
    unhelpful_comments = json.load(read_file)

all_comments = helpful_comments + unhelpful_comments
new_score = [1]*len(helpful_comments) + [0]*len(unhelpful_comments)

new_text = all_comments

#tokenize
input_ids = []
segment_ids = []
attention_masks = []
max_seq_len = 512

for i in tqdm(range(len(new_text))):
    post_title = "[CLS] " + new_text[i] + " [SEP] "
    tokenized_text_title = tokenizer.tokenize(post_title)
    if len(tokenized_text_title) > 512: #restrict the title to 512 tokens including [CLS] and [SEP]
        tokenized_text_title = tokenized_text_title[:511] + [tokenized_text_title[-1]]
    tokenized_text = tokenized_text_title
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_id = [0] * len(tokenized_text_title)
    padding = [0] * (max_seq_len - len(indexed_tokens))
    indexed_tokens += padding
    segments_id += padding
    input_mask = [1] * len(tokenized_text) + padding
    input_ids.append(indexed_tokens)
    segment_ids.append(segments_id)
    attention_masks.append(input_mask)


test_size = 0.1

#split train-test
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, new_score, 
                                                            random_state=42, test_size=test_size)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=42, test_size=test_size)
train_segment_ids, validation_segment_ids, _, _ = train_test_split(segment_ids, input_ids,
                                             random_state=42, test_size=test_size)

#convert data to tensors

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
train_segment_ids = torch.tensor(train_segment_ids)
validation_segment_ids = torch.tensor(validation_segment_ids)

n_gpu = 1
# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = n_gpu * 8

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks,train_segment_ids, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks,validation_segment_ids, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



model.train()

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

# Tracking variables 
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

all_validation_f1 = []
# Put model in evaluation mode to evaluate loss on the validation set
model.eval()

# Tracking variables 
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# Evaluate data for one epoch

all_ground_truth = []
all_predicted = []
for batch in tqdm(validation_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_segment_id, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
    # Forward pass, calculate logit predictions
        logits = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        all_ground_truth += list(label_ids.flatten()) 
        all_predicted += list(np.argmax(logits, axis=1).flatten())

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
all_metrics = metrics.classification_report(all_ground_truth,all_predicted, output_dict= True)
print(all_metrics)
eval_f1 = all_metrics['weighted avg']['f1-score']
print("eval_f1: ", eval_f1)
all_validation_f1.append(eval_f1)
print("all_validation_f1: ", all_validation_f1)
# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
    model.train()
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
  
    # Train the data for one epoch
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_segment_id, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask, labels=b_labels)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training
        train_loss_set.append(loss.item())    
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
    
    
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
  
  
    # Validation
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch

    all_ground_truth = []
    all_predicted = []
    for batch in tqdm(validation_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_segment_id, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask)
    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        all_ground_truth += list(label_ids.flatten()) 
        all_predicted += list(np.argmax(logits, axis=1).flatten())
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    all_metrics = metrics.classification_report(all_ground_truth,all_predicted, output_dict= True)
    print(all_metrics)
    eval_f1 = all_metrics['weighted avg']['f1-score']
    print("eval_f1: ", eval_f1)
    all_validation_f1.append(eval_f1)   
    print("all_validation_f1: ", all_validation_f1)

