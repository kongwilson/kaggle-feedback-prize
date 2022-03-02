"""
TO seek fast implementation with fastai

Copyright (C) Weicong Kong, 23/02/2022
"""
from dotenv import load_dotenv

load_dotenv()
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertForTokenClassification, DistilBertTokenizerFast
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification


pd.options.display.width = 500
pd.options.display.max_columns = 50

RANDOM_SEED = 42
DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
TRANS_PRETRAINED_MODEL_DIR = os.path.join('model_stores', 'longformer-large-4096')
TRAIN_FOLDER = os.path.join(DATA_ROOT, 'train')

train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))


# Label preprocessing
classes = train.discourse_type.unique().tolist()
tags = defaultdict()

for i, c in enumerate(classes):
	tags[f'B-{c}'] = i
	tags[f'I-{c}'] = i + len(classes)
tags[f'O'] = len(classes) * 2
tags[f'Special'] = -100

l2i = dict(tags)

i2l = defaultdict()
for k, v in l2i.items():
	i2l[v] = k
i2l[-100] = 'Special'

i2l = dict(i2l)

N_LABELS = len(i2l) - 1  # not accounting for -100


# Preprocess the raw text
def get_raw_text(ids):
	with open(os.path.join(TRAIN_FOLDER, f'{ids}.txt'), 'r') as file: data = file.read()
	return data


if os.path.exists('train_data_with_raw_text'):
	df = pd.read_csv('train_data_with_raw_text.csv')
else:
	df1 = train.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
	df2 = train.groupby('id')['discourse_start'].apply(list).reset_index(name='starts')
	df3 = train.groupby('id')['discourse_end'].apply(list).reset_index(name='ends')
	df4 = train.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

	df = pd.merge(df1, df2, how='inner', on='id')
	df = pd.merge(df, df3, how='inner', on='id')
	df = pd.merge(df, df4, how='inner', on='id')
	df['text'] = df['id'].apply(get_raw_text)

	df.to_csv('train_data_with_raw_text.csv', index=False)


# Build Training dataset for hugging face model

ds = Dataset.from_pandas(df)
datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

model_checkpoint = os.path.join('model_stores', 'longformer-large-4096')

# WK: `add_prefix_space` Whether or not to add an initial space to the input.
#   This allows to treat the leading word just as any other word
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', add_prefix_space=True)

e = [0, 7, 7, 7, 1, 1, 8, 8, 8, 9, 9, 9, 14, 4, 4, 4]


def fix_beginnings(labels):
	for i in range(1, len(labels)):
		curr_lab = labels[i]
		prev_lab = labels[i - 1]
		if curr_lab in range(7, 14):
			if prev_lab != curr_lab and prev_lab != curr_lab - 7:
				labels[i] = curr_lab - 7
	return labels


fix_beginnings(e)
MAX_LENGTH = 256
STRIDE = 128


## The most important aspect of preprocessing, should use the Kaggle preprocess algo here
def tokenize_and_align_labels(examples):
	o = tokenizer(
		examples['text'], truncation=True, padding=True, return_offsets_mapping=True, max_length=MAX_LENGTH,
		stride=STRIDE, return_overflowing_tokens=True)

	# Since one example might give us several features if it has a long context, we need a map from a feature to
	# its corresponding example. This key gives us just that.
	sample_mapping = o["overflow_to_sample_mapping"]
	# The offset mappings will give us a map from token to character position in the original context. This will
	# help us compute the start_positions and end_positions.
	offset_mapping = o["offset_mapping"]
	attention_mask = o['attention_mask']

	o["labels"] = []

	for i in range(len(offset_mapping)):

		sample_index = sample_mapping[i]

		labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]  # WK: init all labels to be 'pad', for all tokens

		for label_start, label_end, label in \
				list(zip(examples['starts'][sample_index], examples['ends'][sample_index],
				         examples['classlist'][sample_index])):
			for j in range(len(labels)):  # for each of the tokens
				token_start = offset_mapping[i][j][0]
				token_end = offset_mapping[i][j][1]
				if token_start == label_start:
					labels[j] = l2i[f'B-{label}']
				if token_start > label_start and token_end <= label_end:
					labels[j] = l2i[f'I-{label}']

		for k, input_id in enumerate(o['input_ids'][i]):
			# for longformer
			# if input_id in [0, 1, 2]:  # 0: <s> (bos, cls); 1: <pad>; 2: </s> (eos, sep)
			# 	labels[k] = -100
			# for DistilBert
			if input_id in [101, 0, 102]:  # 101: <s> (bos, cls); 0: <pad>; 102: </s> (eos, sep)
				labels[k] = -100

		labels = fix_beginnings(labels)

		o["labels"].append(labels)

	return o


# Processing data in batches
#   the mapped function should accept an input with the format of a slice of the dataset: function(dataset[:10]).
tokenized_datasets = datasets.map(
	tokenize_and_align_labels, batched=True,
	batch_size=4, remove_columns=datasets["train"].column_names)


# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=N_LABELS)
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-23T23:00:40.690854Z","iopub.execute_input":"2021-12-23T23:00:40.693718Z","iopub.status.idle":"2021-12-23T23:00:41.535273Z","shell.execute_reply.started":"2021-12-23T23:00:40.693672Z","shell.execute_reply":"2021-12-23T23:00:41.534215Z"}}
# model_name = model_checkpoint.split(os.path.sep)[-1]
model_name = type(model).__name__
task = 'ner'
# TRAINING HYPERPARAMS
BS = 1
GRAD_ACC = 8
LR = 5e-5
WD = 0.01
WARMUP = 0.1
N_EPOCHS = 5

PATH_TO_SAVE = os.path.join('model_stores', f'{model_name}-feedback_prize')

args = TrainingArguments(
	f"{model_name}-finetuned-{task}",
	evaluation_strategy="epoch",
	logging_strategy="epoch",
	save_strategy="epoch",
	learning_rate=LR,
	per_device_train_batch_size=BS,
	per_device_eval_batch_size=BS,
	num_train_epochs=N_EPOCHS,
	weight_decay=WD,
	# report_to='wandb',
	gradient_accumulation_steps=GRAD_ACC,
	warmup_ratio=WARMUP
)


data_collator = DataCollatorForTokenClassification(tokenizer)


metric = load_metric("seqeval")


def compute_metrics(p):
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	# Remove ignored index (special tokens)
	true_predictions = [
		[i2l[p] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[i2l[l] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]

	results = metric.compute(predictions=true_predictions, references=true_labels)
	return {
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"f1": results["overall_f1"],
		"accuracy": results["overall_accuracy"],
	}


trainer = Trainer(
	model,
	args,
	train_dataset=tokenized_datasets["train"],
	eval_dataset=tokenized_datasets["test"],
	data_collator=data_collator,
	tokenizer=tokenizer,
	compute_metrics=compute_metrics,
)


trainer.train()
