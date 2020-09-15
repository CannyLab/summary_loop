from transformers.modeling_bert import BertForNextSentencePrediction
from transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.nn.modules.loss import CrossEntropyLoss
import torch, os, sys, nltk, tqdm, time, math
from transformers.optimization import AdamW
from utils_logplot import LogPlot
from collections import Counter

STOP_WORDS = set(["'", ".", "!", "?", ",", '"', '-', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', "it's", 'its', 'they', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'or', 'as', 'of', 'at', 'by', 'to', 'not', 'so', "'s", "in", "for", "with", "on"])

class PatternPenalty:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, history_length=30):
        self.stop_words = STOP_WORDS
        self.history_words = []
        self.ngram_history = []
        self.history_length = history_length

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        batch_words = []
        batch_ngrams = []
        for summary in summaries:
            words = nltk.tokenize.word_tokenize(summary.lower())
            gram = 2
            n_grams = [tuple(words[i:(i+gram)]) for i in range(len(words)-gram+1)]

            word_set = set(words)-self.stop_words
            word_set = [w for w in word_set if len(w) > 1]
            self.history_words.append(word_set)
            self.ngram_history.append(n_grams)
            batch_words.append(word_set)
            batch_ngrams.append(n_grams)

        self.history_words = self.history_words[-self.history_length:] # Trim
        self.ngram_history = self.ngram_history[-self.history_length:] # Trim

        word_counter = Counter([w for words in self.history_words for w in words])
        ngram_counter = Counter([ng for ngrams in self.ngram_history for ng in ngrams])

        scores = []
        for words, ngrams in zip(batch_words, batch_ngrams):
            score = 0.0

            if any(word_counter[w] > 0.5*self.history_length for w in words):
                score = 1.0
            if any(ngram_counter[ng] > 0.5*self.history_length for ng in ngrams):
                score = 1.0
                # print(">>>",ngram_counter.most_common(8))
            scores.append(score)
        return scores, None

class LengthPenalty:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, target_length):
        self.target_length = float(target_length)

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        # In lengths, the number of tokens. Is -1 if the summary did not produce an END token, which will be maximum penalty, by design.
        # scores = [1.0-L/self.target_length for L in lengths]
        scores = [1.0 if L > self.target_length else 1.0-L/self.target_length for L in lengths] # This lets it go beyond for free

        return scores, None

class RepeatPenalty:
    # Shouldn't use non-stop words several times in a summary. Fairly constraining.
    def __init__(self):
        self.stop_words = STOP_WORDS

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        scores = []
        for summary in summaries:
            words = nltk.tokenize.word_tokenize(summary.lower())
            L = len(words)
            N_1 = max(2, math.ceil(L / 10.0)) # You shouldn't use the same non-stop word more than 3 times.
            N_2 = math.ceil(L / 8.0)
            word_counts = Counter([w for w in words if w not in self.stop_words])
            all_word_counts = Counter([w for w in words if len(w) > 1])
            if len(word_counts) > 0 and len(all_word_counts) > 0 and (word_counts.most_common(1)[0][1] > N_1 or all_word_counts.most_common(1)[0][1] > N_2):
                # print(L, N_1, N_2)
                # print("Repeat penalty:", word_counts.most_common(3), all_word_counts.most_common(3))
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores, None

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpu_nb", type=int, default=3, help="Which GPU to use. For now single GPU.")
#     parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
#     parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
#     parser.add_argument("--do_train", action='store_true', help="Whether to do some training.")

#     args = parser.parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(args.gpu_nb)

#     print("Loading model")
#     fluency = FluencyCoLA(args.device, model_file="/home/phillab/models/news_gpt2_bs32.bin")

#     if args.do_train:
#         dataloader = fluency.get_training_dataset(args.train_batch_size)
#         param_optimizer = list(fluency.model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]

#         optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
#         scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=len(dataloader))
#         logplot = LogPlot("/home/phillab/logs/fluency/bert_cola_gpu.log")

#         time_save = time.time()
#         optim_every = 4

#         for epi in range(1):
#             print("Epoch", epi)
#             for ib, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
#                 batch = tuple(t.to(args.device) for t in batch)
#                 input_ids, masks, token_types, labels = batch
#                 outputs, = fluency.get_output(input_ids, masks, token_types)

#                 cross_ent = CrossEntropyLoss()
#                 loss = cross_ent(outputs, labels)
#                 acc = torch.argmax(outputs,dim=1).eq(labels).float().mean().item()

#                 loss.backward()

#                 if ib%optim_every == 0:
#                     scheduler.step()  # Update learning rate schedule
#                     optimizer.step()
#                     optimizer.zero_grad()

#                 logplot.cache({"loss": loss.item(), "accuracy": acc}, prefix="T_")
#                 if time.time()-time_save > 60.0:
#                     logplot.save(printing=True)
#                     time_save = time.time()
#                     fluency.save_model("/home/phillab/models/bert_fluency_cola_b.bin")

if __name__ == "__main__":

    summary = "India's Telecom Commission is seeking clarity on 2G spectrum auction issues including the auction"
    summary = "The 39-year-old French star of the silent comedy The Artist scooped the Best Actor statue at the Academy Awards in"
    summary = "The two available units cost $574,000 and $649,900."

    reppen = RepeatPenalty()
    print(reppen.score([summary], [""]))
