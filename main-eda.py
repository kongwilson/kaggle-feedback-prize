"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/02/2022
"""
import os

import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer


TRAIN_DATA_PATH = r'C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021\train.csv'

train_df = pd.read_csv(TRAIN_DATA_PATH)

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

model_name = 'longformer'
tokenizer = AutoTokenizer.from_pretrained(model_name)


# build customised tokenizers


# NER named entity recognition
