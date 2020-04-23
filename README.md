# Summarization

This repository groups the code to train a summarizer model without Summary supervision.

## Training procedure

First need to have three models ready & pre-trained:
 - Coverage model, based on a BERT model, finetuned using the `pretrain_coverage.py` script. A keyword_extrator model is required as well. You can ask me for my file for standard BERT vocab or use the training script in `coverage.py` to make a keyword_extractor for your own vocab.
 - Fluency model, based on a GPT2 model, can use GPT2 directly or a finetuned version using `train_generator.py` (recommended finetuning on domain of summaries, such as news, legal, etc.)
 - Summarizer mode,  based on a GPT2 model. Should use a GPT2 model finetuned to copy at first (using `train_generator.py --task copy`). The copy finetuning is recommended to teach the model to use the <END> token.

Once the three model initializations are ready, the main training script can be run: `train_summarizer.py`. This script outputs a log file with 1 example / minute of summaries produced.
 
## Using Scorer models separately

The Coverage and Fluency model scores can be used separately for comparison. They are respectively in `coverage.py` and `fluency.py`, each model is implemented as a class with a `score(document, summary)` function.
Examples of how to run each model are included in the class files, at the bottom of the files.

# Obtaining the datasets & models

Contact me at phillab@berkeley.edu to obtain:
- Datasets used for training (for now a large corpus of news articles).
- Pretrained models:
   - Coverage model & Keyword Extractor
   - Fluency model (GPT2 finetuned on news)
   - Initial Summarizer (finetuned to copy)
- Final Summarization models (three models: target length 10, 24, 45).
