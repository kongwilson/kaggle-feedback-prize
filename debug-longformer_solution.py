"""
To walk through some longformer solvers

Copyright (C) Weicong Kong, 22/02/2022
"""

import gc

import os
import sys


import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from transformers import AutoConfig, AutoModel, AutoTokenizer

gc.enable()

target_id_map = {
	"B-Lead": 0,
	"I-Lead": 1,
	"B-Position": 2,
	"I-Position": 3,
	"B-Evidence": 4,
	"I-Evidence": 5,
	"B-Claim": 6,
	"I-Claim": 7,
	"B-Concluding Statement": 8,
	"I-Concluding Statement": 9,
	"B-Counterclaim": 10,
	"I-Counterclaim": 11,
	"B-Rebuttal": 12,
	"I-Rebuttal": 13,
	"O": 14,
	"PAD": -100,
}

id_target_map = {v: k for k, v in target_id_map.items()}


def prepare_test_data(test_df, tokenizer):
	ids = test_df["id"].unique()
	test_samples = []
	for fid in ids:
		filename = os.path.join(DATA_ROOT, "test", fid + ".txt")
		with open(filename, "r") as f:
			text = f.read()

		encoded_text = tokenizer.encode_plus(
			text,
			add_special_tokens=False,
			return_offsets_mapping=True,
		)
		input_ids = encoded_text["input_ids"]
		offset_mapping = encoded_text["offset_mapping"]

		sample = {
			"id": fid,
			"input_ids": input_ids,
			"text": text,
			"offset_mapping": offset_mapping,
		}

		test_samples.append(sample)
	return test_samples


class Collate:

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		output = dict()
		output["ids"] = [sample["ids"] for sample in batch]
		output["mask"] = [sample["mask"] for sample in batch]

		# calculate max token length of this batch
		batch_max = max([len(ids) for ids in output["ids"]])

		# add padding
		if self.tokenizer.padding_side == "right":
			output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
			output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
		else:
			output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
			output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

		# convert to tensors
		output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
		output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

		return output


class FeedbackDataset:

	def __init__(self, samples, max_len, tokenizer):
		self.samples = samples
		self.max_len = max_len
		self.tokenizer = tokenizer
		self.length = len(samples)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		input_ids = self.samples[idx]["input_ids"]
		# print(input_ids)
		# print(input_labels)

		# add start token id to the input_ids
		input_ids = [self.tokenizer.cls_token_id] + input_ids

		if len(input_ids) > self.max_len - 1:
			input_ids = input_ids[: self.max_len - 1]

		# add end token id to the input_ids
		input_ids = input_ids + [self.tokenizer.sep_token_id]
		attention_mask = [1] * len(input_ids)

		return {
			"ids": input_ids,
			"mask": attention_mask,
		}


class FeedbackModel(tez.Model):
	def __init__(self, model_name, num_labels):
		super().__init__()
		self.model_name = model_name
		self.num_labels = num_labels
		config = AutoConfig.from_pretrained(model_name)

		hidden_dropout_prob: float = 0.1
		layer_norm_eps: float = 1e-7
		config.update(
			{
				"output_hidden_states": True,
				"hidden_dropout_prob": hidden_dropout_prob,
				"layer_norm_eps": layer_norm_eps,
				"add_pooling_layer": False,
			}
		)
		self.transformer = AutoModel.from_config(config)
		self.output = nn.Linear(config.hidden_size, self.num_labels)

	def forward(self, ids, mask):
		transformer_out = self.transformer(ids, mask)
		sequence_output = transformer_out.last_hidden_state
		logits = self.output(sequence_output)
		logits = torch.softmax(logits, dim=-1)
		return logits, 0, {}


# can specify a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
#   using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method
DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
TRANS_PRETRAINED_MODEL_DIR = os.path.join('model_stores', 'longformer-large-4096')
TEZ_MODEL_STORE = os.path.join('model_stores', 'fblongformerlarge1536')
df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
df_ids = df["id"].unique()

tokenizer = AutoTokenizer.from_pretrained(TRANS_PRETRAINED_MODEL_DIR)
test_samples = prepare_test_data(df, tokenizer)
collate = Collate(tokenizer=tokenizer)


MAX_LEN = 4096
raw_preds = []
for fold_ in range(10):
	current_idx = 0
	test_dataset = FeedbackDataset(test_samples, MAX_LEN, tokenizer)

	if fold_ < 5:
		model = FeedbackModel(model_name=TRANS_PRETRAINED_MODEL_DIR, num_labels=len(target_id_map) - 1)
		model.load(os.path.join(TEZ_MODEL_STORE, f"model_{fold_}.bin"), weights_only=True)
		preds_iter = model.predict(test_dataset, batch_size=8, n_jobs=-1, collate_fn=collate)
	else:
		model = FeedbackModel(model_name=args2.model, num_labels=len(target_id_map) - 1)
		model.load(os.path.join(args2.tez_model, f"model_{fold_ - 5}.bin"), weights_only=True)
		preds_iter = model.predict(test_dataset, batch_size=args2.batch_size, n_jobs=-1, collate_fn=collate)

	current_idx = 0

	for preds in preds_iter:
		preds = preds.astype(np.float16)
		preds = preds / 10
		if fold_ == 0:
			raw_preds.append(preds)
		else:
			raw_preds[current_idx] += preds
			current_idx += 1
	torch.cuda.empty_cache()
	gc.collect()
