# kaggle-feedback-prize

## field explanation

- id - ID code for essay response
- discourse_id - ID code for discourse element
- discourse_start - **character position** where discourse element begins in the essay response
- discourse_end - **character position** where discourse element ends in the essay response
- discourse_text - text of discourse element
- discourse_type - classification of discourse element
- discourse_type_num - enumerated class label of discourse element
- predictionstring - the word indices of the training sample, as required for predictions

## data summary

- There can be partial matches, if the correct discourse type is predicted but on a longer or shorter sequence of words
  than specified in the Ground Truth
- Not necessarily all text of an essay is part of a discourse. For example, in '423A1CA112E2.txt', the title of the
  essay is not part of any discourse.

## Approaches

- Sentence Classification: 
  - https://www.kaggle.com/abhishekme19b069/eda-full-classification-pipeline-bert
  - https://www.kaggle.com/julian3833/feedback-baseline-sentence-classifier-0-226
- NER
  - https://www.kaggle.com/chryzal/feedback-prize-2021-pytorch-better-parameters

It turns out that approaching the problem as a NER problem lead to significantly higher score than sentence 
classification in the community.