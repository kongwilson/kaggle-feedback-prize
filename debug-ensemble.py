"""
DESCRIPTION

Copyright (C) Weicong Kong, 3/03/2022
"""
import numpy as np
import pandas as pd
import os
import transformers
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader

pd.options.display.max_columns = 50
pd.options.display.width = 500

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"
MODEL_STORE = os.path.join('model_stores')


train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))


def apply_stratified_kfold_to_train_data():

    df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))

    # WKNOTE: pd.get_dummies - pandas's one-hot encoder
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols].copy()

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1

    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        print(len(trn_), len(val_))
        dfx.loc[val_, "kfold"] = fold

    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    print(df.kfold.value_counts())
    print(df.pivot_table(index='kfold', values='id', aggfunc=lambda x: len(np.unique(x))))
    print(df.pivot_table(index='kfold', columns='discourse_type', values='id', aggfunc=len))
    df.to_csv("train_folds.csv", index=False)
    return df


# load train data
if os.path.exists('train_folds.csv'):
    train_df = pd.read_csv('train_folds.csv')
else:
    train_df = apply_stratified_kfold_to_train_data()

# prepare label, as the label values when the model was trained
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


def prepare_samples(df, tkz: transformers.models.longformer.tokenization_longformer_fast.LongformerTokenizerFast):
    # prepare test data so that they can be processed by the model
    samples = []
    ids = df['id'].unique()
    for idx in ids:
        filename = os.path.join(DATA_ROOT, 'test', f'{idx}.txt')
        with open(filename, 'r') as f:
            text = f.read()

        encoded_text = tkz.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )  # WKNOTE: same effect as `tkz(text, add_special_tokens=False, return_offsets_mapping=True)`
        input_ids = encoded_text["input_ids"]
        offset_mapping = encoded_text["offset_mapping"]
        sample = {
            'id': idx,
            'input_ids': input_ids,
            'text': text,
            'offset_mapping': offset_mapping
        }
        samples.append(sample)
    return samples


class Collate(object):

    def __init__(self, tkz):
        self.tokenizer = tkz

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


class FeedbackModel(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(model_name)

        hidden_dropout_prob: float = 0.18
        layer_norm_eps: float = 17589e-7
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


tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_STORE, 'longformer-large-4096/'))
test_df = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'))
test_samples = prepare_samples(test_df, tokenizer)
collate = Collate(tkz=tokenizer)

MAX_LENGTH = 4096


raw_preds = []
current_idx = 0
test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
model = FeedbackModel(
    model_name=os.path.join('model_stores', 'longformer-large-4096'), num_labels=len(target_id_map) - 1)
model_path = os.path.join('model_stores', 'fblongformerlarge1536', 'model_0.bin')
model_dict = torch.load(model_path)
model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer

data_loader = DataLoader(
    test_dataset, batch_size=8, num_workers=0, collate_fn=collate
)

a_data = next(iter(data_loader))  # WKNOTE: get a sample from an iterable object
pred = model(**a_data)


