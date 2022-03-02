"""
base on
https://www.kaggle.com/abhishek/two-longformers-are-better-than-1

Copyright (C) Weicong Kong, 17/02/2022
"""
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-01-05T18:19:34.765636Z","iopub.execute_input":"2022-01-05T18:19:34.766525Z","iopub.status.idle":"2022-01-05T18:19:43.074135Z","shell.execute_reply.started":"2022-01-05T18:19:34.76641Z","shell.execute_reply":"2022-01-05T18:19:43.072934Z"}}
import gc

gc.enable()

import sys

sys.path.append("tez")

import os

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from transformers import AutoConfig, AutoModel, AutoTokenizer

# %% [code] {"execution":{"iopub.status.busy":"2022-01-05T18:20:25.812387Z","iopub.execute_input":"2022-01-05T18:20:25.812686Z","iopub.status.idle":"2022-01-05T18:20:25.822348Z","shell.execute_reply.started":"2022-01-05T18:20:25.812656Z","shell.execute_reply":"2022-01-05T18:20:25.821316Z"}}
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


class args1:
	input_path = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
	model = os.path.join('model_stores', 'longformer-large-4096')
	tez_model = os.path.join('model_stores', 'fblongformerlarge1536')
	output = "."
	batch_size = 8
	max_len = 4096


class args2:
	input_path = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
	model = os.path.join('model_stores', 'longformer-large-4096')
	tez_model = os.path.join('model_stores', 'tez-fb-large')
	output = "."
	batch_size = 8
	max_len = 4096


# %% [code] {"execution":{"iopub.status.busy":"2022-01-05T18:20:26.277132Z","iopub.execute_input":"2022-01-05T18:20:26.277591Z","iopub.status.idle":"2022-01-05T18:20:26.28869Z","shell.execute_reply.started":"2022-01-05T18:20:26.277556Z","shell.execute_reply":"2022-01-05T18:20:26.287697Z"}}
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

		# padding_length = self.max_len - len(input_ids)
		# if padding_length > 0:
		#     if self.tokenizer.padding_side == "right":
		#         input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
		#         attention_mask = attention_mask + [0] * padding_length
		#     else:
		#         input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
		#         attention_mask = [0] * padding_length + attention_mask

		# return {
		#     "ids": torch.tensor(input_ids, dtype=torch.long),
		#     "mask": torch.tensor(attention_mask, dtype=torch.long),
		# }

		return {
			"ids": input_ids,
			"mask": attention_mask,
		}


# %% [code]
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


# %% [code] {"execution":{"iopub.status.busy":"2022-01-05T18:20:26.502033Z","iopub.execute_input":"2022-01-05T18:20:26.50234Z","iopub.status.idle":"2022-01-05T18:20:26.511671Z","shell.execute_reply.started":"2022-01-05T18:20:26.502308Z","shell.execute_reply":"2022-01-05T18:20:26.510519Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2022-01-05T18:20:26.715129Z","iopub.execute_input":"2022-01-05T18:20:26.715408Z","iopub.status.idle":"2022-01-05T18:20:26.725686Z","shell.execute_reply.started":"2022-01-05T18:20:26.715365Z","shell.execute_reply":"2022-01-05T18:20:26.724696Z"}}
def _prepare_test_data_helper(args, tokenizer, ids):
	test_samples = []
	for idx in ids:
		filename = os.path.join(args.input_path, "test", idx + ".txt")
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
			"id": idx,
			"input_ids": input_ids,
			"text": text,
			"offset_mapping": offset_mapping,
		}

		test_samples.append(sample)
	return test_samples


def prepare_test_data(df, tokenizer, args):
	test_samples = []
	ids = df["id"].unique()
	# ids_splits = np.array_split(ids, 4)
	#
	# results = Parallel(n_jobs=4, backend="multiprocessing")(
	# 	delayed(_prepare_test_data_helper)(args, tokenizer, idx) for idx in ids_splits
	# )
	test_samples = _prepare_test_data_helper(args, tokenizer, ids)
	# for result in results:
	# 	test_samples.extend(result)

	return test_samples


# %% [code]
DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
df_ids = df["id"].unique()

tokenizer = AutoTokenizer.from_pretrained(args1.model)
test_samples = prepare_test_data(df, tokenizer, args1)
collate = Collate(tokenizer=tokenizer)

raw_preds = []
for fold_ in range(10):
	current_idx = 0
	test_dataset = FeedbackDataset(test_samples, args1.max_len, tokenizer)

	if fold_ < 5:
		model = FeedbackModel(model_name=args1.model, num_labels=len(target_id_map) - 1)
		model.load(os.path.join(args1.tez_model, f"model_{fold_}.bin"), weights_only=True)
		preds_iter = model.predict(test_dataset, batch_size=args1.batch_size, n_jobs=-1, collate_fn=collate)
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

# %% [code]
final_preds = []
final_scores = []

for rp in raw_preds:
	pred_class = np.argmax(rp, axis=2)
	pred_scrs = np.max(rp, axis=2)
	for pred, pred_scr in zip(pred_class, pred_scrs):
		pred = pred.tolist()
		pred_scr = pred_scr.tolist()
		final_preds.append(pred)
		final_scores.append(pred_scr)

for j in range(len(test_samples)):
	tt = [id_target_map[p] for p in final_preds[j][1:]]
	tt_score = final_scores[j][1:]
	test_samples[j]["preds"] = tt
	test_samples[j]["pred_scores"] = tt_score


# %% [code]
def jn(pst, start, end):
	return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
	thresh = 1
	idu = oof['id'].unique()
	idc = idu[1]
	eoof = oof[oof['class'] == "Evidence"]
	neoof = oof[oof['class'] != "Evidence"]
	for thresh2 in range(26, 27, 1):
		retval = []
		for idv in idu:
			for c in ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
				'Counterclaim', 'Rebuttal']:
				q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
				if len(q) == 0:
					continue
				pst = []
				for i, r in q.iterrows():
					pst = pst + [-1] + [int(x) for x in r['predictionstring'].split()]
				start = 1
				end = 1
				for i in range(2, len(pst)):
					cur = pst[i]
					end = i
					# if pst[start] == 205:
					#   print(cur, pst[start], cur - pst[start])
					if (cur == -1 and c != 'Evidence') or ((cur == -1) and (
							(pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
						retval.append((idv, c, jn(pst, start, end)))
						start = i + 1
				v = (idv, c, jn(pst, start, end + 1))
				# print(v)
				retval.append(v)
		roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
		roof = roof.merge(neoof, how='outer')
		return roof


# %% [code]
proba_thresh = {
	"Lead": 0.7,
	"Position": 0.55,
	"Evidence": 0.65,
	"Claim": 0.55,
	"Concluding Statement": 0.7,
	"Counterclaim": 0.5,
	"Rebuttal": 0.55,
}

min_thresh = {
	"Lead": 9,
	"Position": 5,
	"Evidence": 14,
	"Claim": 3,
	"Concluding Statement": 11,
	"Counterclaim": 6,
	"Rebuttal": 4,
}

submission = []
for sample_idx, sample in enumerate(test_samples):
	preds = sample["preds"]
	offset_mapping = sample["offset_mapping"]
	sample_id = sample["id"]
	sample_text = sample["text"]
	sample_input_ids = sample["input_ids"]
	sample_pred_scores = sample["pred_scores"]
	sample_preds = []

	if len(preds) < len(offset_mapping):
		preds = preds + ["O"] * (len(offset_mapping) - len(preds))
		sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

	idx = 0
	phrase_preds = []
	while idx < len(offset_mapping):
		start, _ = offset_mapping[idx]
		if preds[idx] != "O":
			label = preds[idx][2:]
		else:
			label = "O"
		phrase_scores = []
		phrase_scores.append(sample_pred_scores[idx])
		idx += 1
		while idx < len(offset_mapping):
			if label == "O":
				matching_label = "O"
			else:
				matching_label = f"I-{label}"
			if preds[idx] == matching_label:
				_, end = offset_mapping[idx]
				phrase_scores.append(sample_pred_scores[idx])
				idx += 1
			else:
				break
		if "end" in locals():
			phrase = sample_text[start:end]
			phrase_preds.append((phrase, start, end, label, phrase_scores))

	temp_df = []
	for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
		word_start = len(sample_text[:start].split())
		word_end = word_start + len(sample_text[start:end].split())
		word_end = min(word_end, len(sample_text.split()))
		ps = " ".join([str(x) for x in range(word_start, word_end)])
		if label != "O":
			if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
				if len(ps.split()) >= min_thresh[label]:
					temp_df.append((sample_id, label, ps))

	temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
	submission.append(temp_df)

submission = pd.concat(submission).reset_index(drop=True)
submission = link_evidence(submission)
submission.to_csv("submission.csv", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-01-05T18:26:47.003019Z","iopub.status.idle":"2022-01-05T18:26:47.004284Z","shell.execute_reply.started":"2022-01-05T18:26:47.003972Z","shell.execute_reply":"2022-01-05T18:26:47.004003Z"}}
submission.head()