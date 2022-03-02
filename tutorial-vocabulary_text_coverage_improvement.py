"""
https://www.kaggle.com/weicongkong/vocabulary-txt-coverage-improvement-80-to-99/edit

Copyright (C) Weicong Kong, 19/02/2022
"""
# %% [markdown]
# # Introduction
#
# The essays for this challenge have a lot of spelling mistakes and punctuations which can cause out of vocabulary errors for tokenizers.
#
# In this notebook, we will see that initially we have around 80% coverage of word tokens compared to the vocabulary of glove embeddings and then we will improve it to beyond 99%.
#
# We wont be using any lemmatization or stemming to achieve this feat. You are welcome to stem as it may help with some words. Lemmatize probably wont help as it depends on a valid word in the first place.
#
# ---
# ## TLDR;
#
# * Get the exported csv file from notebook output. For the words in raw_words, replace them with clean_words_05 from the text you encounter from your training and testing dataframes. This improves the word embeddings as common mistakes have been clarified and count-wise improvement lifts it from 80% current to 99% new.
#
# ---
# ## Disclaimer
#
# All ideas are taken from the below excellent reference and then adopted for our train/test sets. Amazing work on simple ideas to improve the vocabulary.
#
#
# from @christofhenkel
# https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

# %% [markdown]
# ---
# ---
# # Warning
#
# Make sure that if you use anything like below, then dont train on character positions. Because after spelling corrections the character positions will change.
# I have tried to keep the token counts same as before so training on token positions can potentially work.

from gensim.models import KeyedVectors
import pandas as pd
import os
from tqdm.auto import tqdm

tqdm.pandas()
import numpy as np
import re
from nltk.corpus import stopwords

########################################
# if stopwords are not downloaded to your environment
# import nltk
# nltk.download('stopwords')
########################################
import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")

########################################
# if you want to save / load from local vectors
# word_vectors.save('vectors.kv')
# word_vectors = KeyedVectors.load('vectors.kv')
########################################

LOWER_CASE = True
DATA_DIR = r'C:\Users\wkong\IdeaProjects\kaggle_data\feedback-prize-2021'

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))


def read_train_file(currid="423A1CA112E2", curr_dir=os.path.join(DATA_DIR, 'train')):
	with open(os.path.join(curr_dir, "{}.txt".format(currid)), "r") as f:
		filetext = f.read()

	return filetext


from collections import defaultdict

aam_misspell_dict = {'colour': 'color',
	'centre': 'center',
	'favourite': 'favorite',
	'travelling': 'traveling',
	'counselling': 'counseling',
	'theatre': 'theater',
	'cancelled': 'canceled',
	'labour': 'labor',
	'organisation': 'organization',
	'wwii': 'world war 2',
	'citicise': 'criticize',
	"genericname": "someone",
	"driveless": "driverless",
	"canidates": "candidates",
	"electorial": "electoral",
	"genericschool": "school",
	"polution": "pollution",
	"enviorment": "environment",
	"diffrent": "different",
	"benifit": "benefit",
	"schoolname": "school",
	"artical": "article",
	"elctoral": "electoral",
	"genericcity": "city",
	"recieves": "receives",
	"completly": "completely",
	"enviornment": "environment",
	"somthing": "something",
	"everyones": "everyone",
	"oppurtunity": "opportunity",
	"benifits": "benefits",
	"benificial": "beneficial",
	"tecnology": "technology",
	"paragragh": "paragraph",
	"differnt": "different",
	"reist": "resist",
	"probaly": "probably",
	"usuage": "usage",
	"activitys": "activities",
	"experince": "experience",
	"oppertunity": "opportunity",
	"collge": "college",
	"presedent": "president",
	"dosent": "doesnt",
	"propername": "name",
	"eletoral": "electoral",
	"diffcult": "difficult",
	"desicision": "decision"
}

# # Vocabulary Flow (Lower Case)
#
# 1. Read all text from train files and combine together.
txt = []
for i in tqdm(df["id"].unique()):
	txt.append(read_train_file(i))  # 15594 unique docs

# ## Build Initial Vocabulary

from collections import defaultdict

initial_vocab = defaultdict(int)

for i in tqdm(txt, total=len(txt)):
	words = i.split()
	for word in words:
		initial_vocab[word.lower()] += 1

# # Total Vocabulary Words
#
# Length of the vocabulary ~ 101K words right now

print("Total vocabulary including stopwords is : ", len(initial_vocab))

# # Pandas for all heavylifting
#
# * We will use pandas so that all results achieved can be in different columns and you can download the data in the end and use any columns you would like based on your own tokenizer preferences.
#

word_df = pd.DataFrame(initial_vocab.items(),
                       columns=["raw_words", "raw_words_counts"])
print("-" * 80)
print("Displaying Head of the words dataframe")
display(word_df.head())
print("-" * 80)
print("Displaying Tail of the words dataframe")
display(word_df.tail())
print("-" * 80)

# # Exclude Stopwords
#
# * This means that to report coverage of text we will not count stop words in the analysis.
# * It also helps because in the keyed vectors from gensim stop words wont be included
stops = stopwords.words("english")

word_df["is_stop_word"] = word_df["raw_words"].apply(lambda x: 0 if x not in stops else 1)

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:49.380233Z","iopub.execute_input":"2022-01-22T12:22:49.380471Z","iopub.status.idle":"2022-01-22T12:22:49.388803Z","shell.execute_reply.started":"2022-01-22T12:22:49.380444Z","shell.execute_reply":"2022-01-22T12:22:49.387886Z"}}
word_df["is_stop_word"].value_counts()


# %% [markdown]
# # Analysis 1:
# 1) After all lower case words, we see there are 170 stop words detected from train set

# %% [markdown]
# ---
# ---
#
# # Match Vocab
#
# Let us now see how many words have valid entries in the word_vectors
#
# * apply_coverage only checks if a word exists in glove vectors obtained earlier.

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:49.390094Z","iopub.execute_input":"2022-01-22T12:22:49.390341Z","iopub.status.idle":"2022-01-22T12:22:49.605302Z","shell.execute_reply.started":"2022-01-22T12:22:49.390311Z","shell.execute_reply":"2022-01-22T12:22:49.604097Z"}}
def apply_coverage(x):
	if x in word_vectors:
		return 1
	return 0


word_df["raw_in_vectors"] = word_df["raw_words"].apply(apply_coverage)


# %% [markdown]
# * get_coverage checks the amount of vocabulary and text coverage when comparing to glove vectors
# * It creates a new column named column_word_presence to indicate if the target word exists or not.
# * It also displays the coverage statistics and returns a dataframe.
#
#     '''
#         column_words : the column containing the words for which we check coverage
#         column_word_counts : the column containing pre-computed word counts for the words in question
#         column_word_presence : Just an output column name where we will output if word exists in word_vectors
#         exc_stop : Should we exclude stopwrods from coverage analysis or not
#     '''

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:49.607286Z","iopub.execute_input":"2022-01-22T12:22:49.607515Z","iopub.status.idle":"2022-01-22T12:22:49.621412Z","shell.execute_reply.started":"2022-01-22T12:22:49.607487Z","shell.execute_reply":"2022-01-22T12:22:49.620284Z"}}
def get_coverage(column_words,
		column_word_counts,
		column_word_presence,
		df, exc_stop=True):
	'''
		column_words : the column containing the words for which we check coverage
		column_word_counts : the column containing pre-computed word counts for the words in question
		column_word_presence : Just an output column name where we will output if word exists in word_vectors
		exc_stop : Should we exclude stopwrods from coverage analysis or not
	'''
	word_df = df.copy()
	word_df[column_word_presence] = word_df[column_words].apply(apply_coverage)
	print("-" * 80)

	# display(word_df[column_word_presence].value_counts(normalize = True))
	# print("-" * 80)
	if exc_stop == False:
		word_coverage = 100 * word_df[column_word_presence].value_counts(normalize=True)[1]
		text_coverage = 100 * word_df.groupby([column_word_presence])[column_word_counts].sum()[1] / (
					word_df.groupby([column_word_presence])[column_word_counts].sum()[0] +
					word_df.groupby([column_word_presence])[column_word_counts].sum()[1])
	else:
		print("EXCLUDING STOP WORD FROM ANALYSIS...")
		word_coverage = 100 * word_df[word_df["is_stop_word"] == 0][column_word_presence].value_counts(normalize=True)[
			1]
		text_coverage = 100 * \
		                word_df[word_df["is_stop_word"] == 0].groupby([column_word_presence])[column_word_counts].sum()[
			                1] / (word_df[word_df["is_stop_word"] == 0].groupby([column_word_presence])[
				                      column_word_counts].sum()[0] +
		                          word_df[word_df["is_stop_word"] == 0].groupby([column_word_presence])[
				                      column_word_counts].sum()[1])

	if exc_stop:
		print("Total words in {} were {} and {:.2f}% words were found in the word_vectors.".format(column_words,
		                                                                                           len(word_df[word_df[
			                                                                                                       "is_stop_word"] == 0]),
		                                                                                           word_coverage))
	else:
		print("Total words in {} were {} and {:.2f}% words were found in the word_vectors.".format(column_words,
		                                                                                           len(word_df),
		                                                                                           word_coverage))

	print("-" * 80)
	print("From text coverage, {:.2f}% text is coverage in word_vectors.".format(text_coverage))
	print("-" * 80)
	return word_df


# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:49.625341Z","iopub.execute_input":"2022-01-22T12:22:49.625634Z","iopub.status.idle":"2022-01-22T12:22:49.888039Z","shell.execute_reply.started":"2022-01-22T12:22:49.625604Z","shell.execute_reply":"2022-01-22T12:22:49.887016Z"}}
word_df = get_coverage("raw_words", "raw_words_counts", "raw_in_vectors", word_df)


# %% [markdown]
# ---
# ---
# # Analysis 2
#
# * So we have missing vocabulary for around **76%** of the total words
# * In terms of usage frequency, we have coverage of 80% of the words if repetitions are taken into account
#
# ---
# ---
#
# # Objective
#
# * Our objective is to improve the **text coverage** so that the tokenizers can improve their performances
#
#
# **Remember** we DONT WANT TO INCREASE or DECREASE the number of tokens. As this will make training and predictionstring too difficult and cause problems on the submission dataset

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:49.889329Z","iopub.execute_input":"2022-01-22T12:22:49.889552Z","iopub.status.idle":"2022-01-22T12:22:51.151386Z","shell.execute_reply.started":"2022-01-22T12:22:49.889525Z","shell.execute_reply":"2022-01-22T12:22:51.150335Z"}}
def preprocess(x):
	x = x.replace("n't", "nt")

	x = str(x)
	if LOWER_CASE:
		x = x.lower()

	if len(x.strip()) == 1:
		return x  # special case if a punctuation was the only alphabet in the token.

	for punct in "/-'&":
		x = x.replace(punct, '')
	for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
		x = x.replace(punct, '')

	x = re.sub('[0-9]{1,}', '#', x)  # replace all numbers by #
	if len(x.strip()) < 1:
		x = '.'  # if it was all punctuations like ------ or ..... or .;?!!. Then we return only a period to keep token consistent performance.
	return x


word_df["clean_words_01"] = word_df["raw_words"].progress_apply(lambda x: preprocess(x))

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.152605Z","iopub.execute_input":"2022-01-22T12:22:51.15284Z","iopub.status.idle":"2022-01-22T12:22:51.163615Z","shell.execute_reply.started":"2022-01-22T12:22:51.152813Z","shell.execute_reply":"2022-01-22T12:22:51.162944Z"}}
word_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.164778Z","iopub.execute_input":"2022-01-22T12:22:51.165129Z","iopub.status.idle":"2022-01-22T12:22:51.382731Z","shell.execute_reply.started":"2022-01-22T12:22:51.165092Z","shell.execute_reply":"2022-01-22T12:22:51.382083Z"}}
temp = pd.DataFrame(word_df.groupby(["clean_words_01"])["raw_words_counts"].sum()).reset_index()
temp.columns = ["clean_words_01", "clean_words_01_counts"]

word_df = word_df.merge(temp, on=["clean_words_01"], how='left')

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.383842Z","iopub.execute_input":"2022-01-22T12:22:51.384219Z","iopub.status.idle":"2022-01-22T12:22:51.62625Z","shell.execute_reply.started":"2022-01-22T12:22:51.384189Z","shell.execute_reply":"2022-01-22T12:22:51.624941Z"}}
word_df = get_coverage("clean_words_01", "clean_words_01_counts", "clean_01_in_vectors", word_df)

# %% [markdown]
# ---
# # Analysis #3
#
# * **Amazing**, we improved the text coverage from **80**% to **99.76**%
# * Lets try to do more
# ---

# %% [markdown]
# # Accented alphabet replacements
#
# * á to a and so on....

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.628221Z","iopub.execute_input":"2022-01-22T12:22:51.628535Z","iopub.status.idle":"2022-01-22T12:22:51.644699Z","shell.execute_reply.started":"2022-01-22T12:22:51.628495Z","shell.execute_reply":"2022-01-22T12:22:51.644037Z"}}
# Reference:
# https://itqna.net/questions/9818/how-remove-accented-expressions-regular-expressions-python
import re

# char codes: https://unicode-table.com/en/#basic-latin
accent_map = {
	u'\u00c0': u'A',
	u'\u00c1': u'A',
	u'\u00c2': u'A',
	u'\u00c3': u'A',
	u'\u00c4': u'A',
	u'\u00c5': u'A',
	u'\u00c6': u'A',
	u'\u00c7': u'C',
	u'\u00c8': u'E',
	u'\u00c9': u'E',
	u'\u00ca': u'E',
	u'\u00cb': u'E',
	u'\u00cc': u'I',
	u'\u00cd': u'I',
	u'\u00ce': u'I',
	u'\u00cf': u'I',
	u'\u00d0': u'D',
	u'\u00d1': u'N',
	u'\u00d2': u'O',
	u'\u00d3': u'O',
	u'\u00d4': u'O',
	u'\u00d5': u'O',
	u'\u00d6': u'O',
	u'\u00d7': u'x',
	u'\u00d8': u'0',
	u'\u00d9': u'U',
	u'\u00da': u'U',
	u'\u00db': u'U',
	u'\u00dc': u'U',
	u'\u00dd': u'Y',
	u'\u00df': u'B',
	u'\u00e0': u'a',
	u'\u00e1': u'a',
	u'\u00e2': u'a',
	u'\u00e3': u'a',
	u'\u00e4': u'a',
	u'\u00e5': u'a',
	u'\u00e6': u'a',
	u'\u00e7': u'c',
	u'\u00e8': u'e',
	u'\u00e9': u'e',
	u'\u00ea': u'e',
	u'\u00eb': u'e',
	u'\u00ec': u'i',
	u'\u00ed': u'i',
	u'\u00ee': u'i',
	u'\u00ef': u'i',
	u'\u00f1': u'n',
	u'\u00f2': u'o',
	u'\u00f3': u'o',
	u'\u00f4': u'o',
	u'\u00f5': u'o',
	u'\u00f6': u'o',
	u'\u00f8': u'0',
	u'\u00f9': u'u',
	u'\u00fa': u'u',
	u'\u00fb': u'u',
	u'\u00fc': u'u'
}


def accent_remove(m):
	return accent_map[m.group(0)]


string_velha = "Olá você está ????   "
string_nova = re.sub(u'([\u00C0-\u00FC])', accent_remove, string_velha.encode().decode('utf-8'))
string_nova

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.64592Z","iopub.execute_input":"2022-01-22T12:22:51.646388Z","iopub.status.idle":"2022-01-22T12:22:51.875335Z","shell.execute_reply.started":"2022-01-22T12:22:51.646337Z","shell.execute_reply":"2022-01-22T12:22:51.874425Z"}}
word_df["clean_words_02"] = word_df["clean_words_01"].apply(lambda x: re.sub(u'([\u00C0-\u00FC])',
                                                                             accent_remove,
                                                                             x.encode().decode('utf-8'))
                                                            )

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:51.876697Z","iopub.execute_input":"2022-01-22T12:22:51.87697Z","iopub.status.idle":"2022-01-22T12:22:52.099607Z","shell.execute_reply.started":"2022-01-22T12:22:51.876935Z","shell.execute_reply":"2022-01-22T12:22:52.098987Z"}}
temp = pd.DataFrame(word_df.groupby(["clean_words_02"])["raw_words_counts"].sum()).reset_index()
temp.columns = ["clean_words_02", "clean_words_02_counts"]

word_df = word_df.merge(temp, on=["clean_words_02"], how='left')

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:52.100622Z","iopub.execute_input":"2022-01-22T12:22:52.101376Z","iopub.status.idle":"2022-01-22T12:22:52.407008Z","shell.execute_reply.started":"2022-01-22T12:22:52.101342Z","shell.execute_reply":"2022-01-22T12:22:52.406137Z"}}
word_df = get_coverage("clean_words_02", "clean_words_02_counts", "clean_02_in_vectors", word_df)

# %% [markdown]
# # Analysis 4
#
# * Seems that accented character changes did not improve score too much. Lets see some examples where these replacements were made

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:52.408463Z","iopub.execute_input":"2022-01-22T12:22:52.408921Z","iopub.status.idle":"2022-01-22T12:22:52.425957Z","shell.execute_reply.started":"2022-01-22T12:22:52.408885Z","shell.execute_reply":"2022-01-22T12:22:52.425312Z"}}
word_df[word_df["clean_words_01_counts"] != word_df["clean_words_02_counts"]].head(10)

# %% [markdown]
# ---
# ---
#
# # Misspellings
#
# * I created a dictionary in the beginning of notebook **aam_misspell_dict** to include common errors that I can see. You can improve upon it
# * There are definitely a lot of **misspellings** at work
# * There are also anonymous names playing. (like **genericschool**, **genericname** etc..)
# * Seeing that mainly there are a **limited number of topics**, we can perform some basic spelling corrections on the essay topics
#

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:52.427005Z","iopub.execute_input":"2022-01-22T12:22:52.427329Z","iopub.status.idle":"2022-01-22T12:22:52.466035Z","shell.execute_reply.started":"2022-01-22T12:22:52.427303Z","shell.execute_reply":"2022-01-22T12:22:52.465378Z"}}
word_df["clean_words_03"] = word_df["clean_words_02"].apply(
	lambda x: x if x not in aam_misspell_dict else aam_misspell_dict[x])

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:52.467696Z","iopub.execute_input":"2022-01-22T12:22:52.46882Z","iopub.status.idle":"2022-01-22T12:22:52.70114Z","shell.execute_reply.started":"2022-01-22T12:22:52.468747Z","shell.execute_reply":"2022-01-22T12:22:52.700016Z"}}
temp = pd.DataFrame(word_df.groupby(["clean_words_03"])["raw_words_counts"].sum()).reset_index()
temp.columns = ["clean_words_03", "clean_words_03_counts"]

word_df = word_df.merge(temp, on=["clean_words_03"], how='left')

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:52.703892Z","iopub.execute_input":"2022-01-22T12:22:52.704306Z","iopub.status.idle":"2022-01-22T12:22:53.037998Z","shell.execute_reply.started":"2022-01-22T12:22:52.704263Z","shell.execute_reply":"2022-01-22T12:22:53.037Z"}}
word_df = get_coverage("clean_words_03", "clean_words_03_counts", "clean_03_in_vectors", word_df)

# %% [markdown]
# # Viola
#
# * Another improvement from **99.76%** to **99.84%**
# * Can we do better ???

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.03951Z","iopub.execute_input":"2022-01-22T12:22:53.039951Z","iopub.status.idle":"2022-01-22T12:22:53.079343Z","shell.execute_reply.started":"2022-01-22T12:22:53.039914Z","shell.execute_reply":"2022-01-22T12:22:53.078418Z"}}
word_df[word_df["clean_03_in_vectors"] == 0].sort_values(by=["clean_words_03_counts"], ascending=False).head(20)

# %% [markdown]
# # Analysis #5
#
# * **Shouldnt** is giving us some problems. Well it shouln't (pun-intended)
# * The vocabulary contains should and not separately but I dont want to increase number of tokens, so we will replace all shouldnt with **shant** which has similar meaning

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.080987Z","iopub.execute_input":"2022-01-22T12:22:53.081963Z","iopub.status.idle":"2022-01-22T12:22:53.087295Z","shell.execute_reply.started":"2022-01-22T12:22:53.081911Z","shell.execute_reply":"2022-01-22T12:22:53.086239Z"}}
aam_misspell_dict.update({"shouldnt": "shant"})

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.088828Z","iopub.execute_input":"2022-01-22T12:22:53.089734Z","iopub.status.idle":"2022-01-22T12:22:53.412335Z","shell.execute_reply.started":"2022-01-22T12:22:53.089678Z","shell.execute_reply":"2022-01-22T12:22:53.411353Z"}}
word_df["clean_words_04"] = word_df["clean_words_03"].apply(
	lambda x: x if x not in aam_misspell_dict else aam_misspell_dict[x])
temp = pd.DataFrame(word_df.groupby(["clean_words_04"])["raw_words_counts"].sum()).reset_index()
temp.columns = ["clean_words_04", "clean_words_04_counts"]

word_df = word_df.merge(temp, on=["clean_words_04"], how='left')

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.413663Z","iopub.execute_input":"2022-01-22T12:22:53.413917Z","iopub.status.idle":"2022-01-22T12:22:53.710645Z","shell.execute_reply.started":"2022-01-22T12:22:53.413887Z","shell.execute_reply":"2022-01-22T12:22:53.709716Z"}}
word_df = get_coverage("clean_words_04", "clean_words_04_counts", "clean_04_in_vectors", word_df)

# %% [markdown]
# ---
# ---
#
# # Wow
#
# * Another 0.01% improvement
# * Lets see the top words still giving us issues

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.712226Z","iopub.execute_input":"2022-01-22T12:22:53.712475Z","iopub.status.idle":"2022-01-22T12:22:53.750182Z","shell.execute_reply.started":"2022-01-22T12:22:53.712445Z","shell.execute_reply":"2022-01-22T12:22:53.749265Z"}}
word_df[word_df["clean_04_in_vectors"] == 0].sort_values(by=["clean_words_04_counts"], ascending=False)[:10]

# %% [markdown]
# ---
# ---
#
# # Analysis 6 - Risky Choices Ahead
#
# * Now we can see that the vocabulary with most trouble is sort of problem-specific and not general enough
# * We can replace the **studentdesigned** or **teacherdesigned** as designed.. This would probably change the contextual meaning but can improve tokenize performance as well.
# * I will replace **studentname** as **myself**
# * I will replace **teachername** as **teacher**
# * I will replace **winnertakeall** as **winner-take-all**

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.751542Z","iopub.execute_input":"2022-01-22T12:22:53.751808Z","iopub.status.idle":"2022-01-22T12:22:53.75694Z","shell.execute_reply.started":"2022-01-22T12:22:53.751755Z","shell.execute_reply":"2022-01-22T12:22:53.755835Z"}}
aam_misspell_dict.update({"teacherdesigned": "designed",
	                         "studentname": "myself",
	                         "studentdesigned": "designed",
	                         "teachername": "teacher",
	                         "winnertakeall": "winner-take-all"})

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:53.758372Z","iopub.execute_input":"2022-01-22T12:22:53.75869Z","iopub.status.idle":"2022-01-22T12:22:54.158875Z","shell.execute_reply.started":"2022-01-22T12:22:53.758659Z","shell.execute_reply":"2022-01-22T12:22:54.157862Z"}}
word_df["clean_words_05"] = word_df["clean_words_04"].apply(
	lambda x: x if x not in aam_misspell_dict else aam_misspell_dict[x])
temp = pd.DataFrame(word_df.groupby(["clean_words_05"])["raw_words_counts"].sum()).reset_index()
temp.columns = ["clean_words_05", "clean_words_05_counts"]

word_df = word_df.merge(temp, on=["clean_words_05"], how='left')

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:54.161013Z","iopub.execute_input":"2022-01-22T12:22:54.16138Z","iopub.status.idle":"2022-01-22T12:22:54.499044Z","shell.execute_reply.started":"2022-01-22T12:22:54.161333Z","shell.execute_reply":"2022-01-22T12:22:54.498068Z"}}
word_df = get_coverage("clean_words_05", "clean_words_05_counts", "clean_05_in_vectors", word_df)

# %% [markdown]
# ---
# ---
# # Conclusion
#
# * After application of several transformations, we have created a dictionary which improves the text-coverage from **80%** to **99.92%**
# * Hopefully this can improve the prediction performance from different models.
# * Do post critique/feedback
#
# * You can use the exported dataframe to create / use as dictionary for your tokens.

# %% [code] {"execution":{"iopub.status.busy":"2022-01-22T12:22:54.500422Z","iopub.execute_input":"2022-01-22T12:22:54.500645Z","iopub.status.idle":"2022-01-22T12:22:55.332218Z","shell.execute_reply.started":"2022-01-22T12:22:54.500618Z","shell.execute_reply":"2022-01-22T12:22:55.331446Z"}}
print("Exporting the created dictionary now. ")
word_df.to_csv("cleaned_word_dict.csv")

# %% [markdown]
# Have a nice day fellows. Happy kaggling!

# %% [code]
