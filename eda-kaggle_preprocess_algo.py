"""
Chris' finding: https://www.kaggle.com/c/feedback-prize-2021/discussion/297591

Official response: https://www.kaggle.com/c/feedback-prize-2021/discussion/297688

Copyright (C) Weicong Kong, 25/02/2022
"""
import os.path

import pandas as pd

DATA_ROOT = r"C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021"

train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))

train_df['text_len'] = train_df['discourse_text'].apply(lambda x: len(x.split()))
train_df['pred_len'] = train_df['predictionstring'].apply(lambda x: len(x.split()))

mismatched_df = train_df[train_df['text_len'] != train_df['pred_len']].sort_values('id')

mismatched_sample = mismatched_df.loc[74057]
mismatched_sample_path = os.path.join(DATA_ROOT, 'train', f'{mismatched_sample["id"]}.txt')
with open(mismatched_sample_path, 'r') as f:
	mismatched_sample_text = f.read()

print(mismatched_sample_text)
print(mismatched_sample['discourse_text'])
print(mismatched_sample['predictionstring'])
print(mismatched_sample_text[int(mismatched_sample['discourse_start'])])

words = mismatched_sample_text.split()


# kaggle algo to solve the mismatching
char_start = int(mismatched_sample['discourse_start'])
char_end = int(mismatched_sample['discourse_end'])
print(mismatched_sample_text[char_start], mismatched_sample_text[char_end])
word_start = len(mismatched_sample_text[:char_start].split())
word_end = word_start + len(mismatched_sample_text[char_start:char_end].split())
word_end = min(word_end, len(mismatched_sample_text.split()))
predictionstring = " ".join([str(x) for x in range(word_start, word_end)])
print(predictionstring)
print(mismatched_sample['predictionstring'])
pred_words = list(map(lambda x: mismatched_sample_text.split()[int(x)], predictionstring.split()))
print(' '.join(pred_words))
print('The discourse text in the training is: ')
print(mismatched_sample['discourse_text'])


def calc_word_indices(full_text, discourse_start, discourse_end):
	start_index = len(full_text[:discourse_start].split())
	token_len = len(full_text[discourse_start:discourse_end].split())
	output = list(range(start_index, start_index + token_len))
	if output[-1] >= len(full_text.split()):
		output = list(range(start_index, start_index + token_len-1))
	return output


word_indices = calc_word_indices(mismatched_sample_text, char_start, char_end)