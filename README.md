# Summary Loop

This repository contains the code to apply the Summary Loop procedure to train a Summarizer in an unsupervised way, without example summaries.

<p align="center">
  <img width="460" height="300" src="https://people.eecs.berkeley.edu/~phillab/images/summary_loop.png">
</p>

## Training Procedure

We provide pre-trained models for each component needed in the Summary Loop Release:

- `keyword_extractor.joblib`: An sklearn pipeline that will extract can be used to compute tf-idf scores of words according to the BERT vocabulary, which is used by the Masking Procedure,
- `bert_coverage.bin`: A bert-base-uncased finetuned model on the task of Coverage for the news domain,
- `fluency_news_bs32.bin`: A GPT2 (base) model finetuned on a large corpus of news articles, used as the Fluency model,
- `gpt2_copier23.bin`: A GPT2 (base) model that can be used as an initial point for the Summarizer model.

We also provide:
- `pretrain_coverage.py` script to train a coverage model from scratch, 
- `train_generator.py` to train a fluency model from scratch (we recommend Fluency model on domain of summaries, such as news, legal, etc.)

Once all the pretraining models are ready, training a summarizer can be done using the `train_summary_loop.py`:
```
python train_summary_loop.py --experiment wikinews_test --dataset_file data/wikinews.db
```

## Scorer Models

The Coverage and Fluency model scores can be used separately for analysis, evaluation, etc.
They are respectively in `coverage.py` and `fluency.py`, each model is implemented as a class with a `score(document, summary)` function.
Examples of how to run each model are included in the class files, at the bottom of the files.

## Further Questions

Feel free to contact me at phillab@berkeley.edu to discuss the results, the code or future steps.
