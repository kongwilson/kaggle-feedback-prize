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

## Notes on winning solutions 

### Problem modeling

- token classification -> lgb sentence classification [1st place](https://www.kaggle.
  com/competitions/feedback-prize-2021/discussion/313177)
- 

### Hyperparameters

- epochs: 7 at learning rates of 2.5e-5 for 3 epochs, 2.5e-6 for 3 epochs and 2.5e-7 for the last epoch
- learning rates: 2.5e-5 ~ 2.5e-7 (for longformer and deberta)

### CV Setup

- should monitor whether the local cross-validation would be consistent with public leaderboard


### Ensemble Strategy

[3rd place](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313235)

- ensemble of `Longformer` and  `deberta-xl` 
  - `deberta-xl` outperforms `longformer` by quite a bit, although it's more difficult to train, because an input 
    sequence cannot be longer than 512, while longformer can handle upto 4096
  - sliding window (SW) to train Deberta-xl
- Augmentation
  - **Masked aug**, where 15% of the tokens was masked during training
  - **Cutmix**, similar to how cutmix works for images, we cut a portion of one sequence and paste it (and its 
    labels) into another sequence in the same batch
- 2nd place hugging face models:
  - microsoft/deberta-large
  - microsoft/deberta-large + lstm
  - microsoft/deberta-v3-large
  - microsoft/deberta-xlarge
  - **microsoft/deberta-v2-xlarge**
  - **allenai/longformer-large-4096**
  - [LSG converted roberta](https://github.com/ccdv-ai/convert_checkpoint_to_lsg)
  - funnel-transformer/large
  - google/bigbird-roberta-base
  - uw-madison/yoso-4096
  - **roberta-large**
  - distibart-mnli-12-9
  - bart-large-finetuned-squadv1

### Post processing

- **1st place**: 
  - lower the threshold of positive from stage 1 deep learning output to have _as many candidate 
    samples as possible_ to start with (aiming for higher **recall**)
  - 