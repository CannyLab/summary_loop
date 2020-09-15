# Summary Loop

This repository contains the code for ACL2020 paper: [The Summary  Loop: Learning to Write Abstractive Summaries Without Examples](http://people.ischool.berkeley.edu/~hearst/papers/Laban_ACL2020_Abstractive_Summarization.pdf).

<p align="center">
  <img width="460" height="300" src="https://people.eecs.berkeley.edu/~phillab/images/summary_loop.png">
</p>

## Training Procedure

We provide pre-trained models for each component needed in the [Summary Loop Release](https://github.com/CannyLab/summary_loop/releases/tag/v0.1):

- `keyword_extractor.joblib`: An sklearn pipeline that will extract can be used to compute tf-idf scores of words according to the BERT vocabulary, which is used by the Masking Procedure,
- `bert_coverage.bin`: A bert-base-uncased finetuned model on the task of Coverage for the news domain,
- `fluency_news_bs32.bin`: A GPT2 (base) model finetuned on a large corpus of news articles, used as the Fluency model,
- `gpt2_copier23.bin`: A GPT2 (base) model that can be used as an initial point for the Summarizer model.

[In the release](https://github.com/CannyLab/summary_loop/releases/tag/v0.1), we also provide:
- `pretrain_coverage.py` script to train a coverage model from scratch, 
- `train_generator.py` to train a fluency model from scratch (we recommend Fluency model on domain of summaries, such as news, legal, etc.)

Once all the pretraining models are ready, training a summarizer can be done using the `train_summary_loop.py`:
```
python train_summary_loop.py --experiment wikinews_test --dataset_file data/wikinews.db
```

## Scorer Models

The Coverage and Fluency model and Guardrails scores can be used separately for analysis, evaluation, etc.
They are respectively in `model_coverage.py` and `model_guardrails.py`, each model is implemented as a class with a `score(document, summary)` function.
The Fluency model is a Language model, which is also the generator (in `model_generator.py`).
Examples of how to run each model are included in the class files, at the bottom of the files.

## Bringing in your own data

Want to test out the Summary Loop on a different language/type of text?
A [Jupyter Notebook](https://github.com/CannyLab/summary_loop/blob/master/Dataset%20SQLite3%20Example.ipynb) can help you bring your own data into the SQLite format we use in the pre-training scripts. Otherwise you can modify the scripts' data loading (`DataLoader`) and collate function (`collate_fn`).

## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:
```
@inproceedings{laban2020summary,
  title={The Summary Loop: Learning to Write Abstractive Summaries Without Examples},
  author={Laban, Philippe and Hsi, Andrew and Canny, John and Hearst, Marti A},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  volume={1},
  year={2020}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, you can contact us at phillab@berkeley.edu.
All contributions welcome! For example, if you have a type of text data on which you want to apply the Summary Loop.
