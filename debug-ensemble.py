"""
DESCRIPTION

Copyright (C) Weicong Kong, 3/03/2022
"""
import gc
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

gc.enable()

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


def prepare_samples(
        df, is_train: bool, tkz: transformers.models.longformer.tokenization_longformer_fast.LongformerTokenizerFast):
    # prepare test data so that they can be processed by the model
    samples = []
    ids = df['id'].unique()
    train_or_test = 'train' if is_train else 'test'
    for idx in ids:
        filename = os.path.join(DATA_ROOT, train_or_test, f'{idx}.txt')
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


def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26,27, 1):
        retval = []
        for idv in idu:
            for c in  ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                   'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i,r in q.iterrows():
                    pst = pst +[-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2,len(pst)):
                    cur = pst[i]
                    end = i
                    #if pst[start] == 205:
                    #   print(cur, pst[start], cur - pst[start])
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                #print(v)
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring'])
        roof = roof.merge(neoof, how='outer')
        return roof


tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_STORE, 'longformer-large-4096/'))
test_df = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'))
test_samples = prepare_samples(test_df, is_train=False, tkz=tokenizer)
train_samples = prepare_samples(train_df, is_train=True, tkz=tokenizer)
collate = Collate(tkz=tokenizer)

MAX_LENGTH = 4096


raw_preds = []
current_idx = 0
test_dataset = FeedbackDataset(test_samples, MAX_LENGTH, tokenizer)
train_dataset = FeedbackDataset(train_samples, MAX_LENGTH, tokenizer)
model = FeedbackModel(
    model_name=os.path.join('model_stores', 'longformer-large-4096'), num_labels=len(target_id_map) - 1)
model_path = os.path.join('model_stores', 'fblongformerlarge1536', 'model_0.bin')
model_dict = torch.load(model_path)
model.load_state_dict(model_dict)  # this loads the nn.Module and match all the parameters in model.transformer

test_data_loader = DataLoader(
    test_dataset, batch_size=8, num_workers=0, collate_fn=collate
)
train_data_loader = DataLoader(
    train_dataset, batch_size=8, num_workers=0, collate_fn=collate
)

a_data = next(iter(train_data_loader))  # WKNOTE: get a sample from an iterable object
model.eval()
with torch.no_grad():
    pred = model(**a_data)


pred_prob = pred[0]
pred_prob =pred_prob.cpu().detach().numpy()
pred_class = np.argmax(pred_prob, axis=2)
pred_scrs = np.max(pred_prob, axis=2)


train_debug_samples = train_samples[:8]
for j in range(len(train_debug_samples)):
    tt = [id_target_map[p] for p in pred_class[j][1:]]
    tt_score = pred_scrs[j][1:]
    train_debug_samples[j]['preds'] = tt
    train_debug_samples[j]['pred_scores'] = tt_score


proba_thresh = {
    "Lead": 0.687,
    "Position": 0.537,
    "Evidence": 0.637,
    "Claim": 0.537,
    "Concluding Statement": 0.687,
    "Counterclaim": 0.537,
    "Rebuttal": 0.537,
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


def build_feedback_prediction_string(
        preds, sample_pred_scores, sample_id, sample_text, offset_mapping):
    if len(preds) < len(offset_mapping):
        preds = preds + ["O"] * (len(offset_mapping) - len(preds))
        sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

    # loop through each sub-token and construct the LABELLED discourse segments
    idx = 0
    discourse_labels_details = []
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
            discourse = sample_text[start:end]
            discourse_labels_details.append((discourse, start, end, label, phrase_scores))

    temp_df = []
    for phrase_idx, (discourse, start, end, label, phrase_scores) in enumerate(discourse_labels_details):
        word_start = len(sample_text[:start].split())
        word_end = word_start + len(sample_text[start:end].split())
        word_end = min(word_end, len(sample_text.split()))
        ps = " ".join([str(x) for x in range(word_start, word_end)])
        if label != "O":
            if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                if len(ps.split()) >= min_thresh[label]:
                    temp_df.append((sample_id, label, ps))
    temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])
    return temp_df


for sample_idx, sample in enumerate(train_debug_samples):
    preds = sample["preds"]
    offset_mapping = sample["offset_mapping"]
    sample_id = sample["id"]
    sample_text = sample["text"]
    sample_pred_scores = sample["pred_scores"]
    sample_preds = []

    temp_df = build_feedback_prediction_string(preds, sample_pred_scores, sample_id, sample_text, offset_mapping)
    submission.append(temp_df)

submission = pd.concat(submission).reset_index(drop=True)
submission = link_evidence(submission)