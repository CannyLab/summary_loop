from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler

import tqdm, nltk, torch, time, numpy as np, argparse, os
from utils_logplot import LogPlot
from model_coverage import KeywordCoverage
from utils_dataset import SQLDataset
import utils_misc 

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--dataset_file", type=str, required=True, help="Which dataset file to use. Can be full path or the root folder will be attached.")

parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
parser.add_argument("--n_kws", type=int, default=15, help="Top n words (tf-idf wise) will be masked in the coverage model.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

models_folder = "/home/ubuntu/models/"
logs_folder =   "/home/ubuntu/logs/"


args = parser.parse_args()

if args.device == "cuda":
    freer_gpu = str(utils_misc.get_freer_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
    args.experiment += "_"+freer_gpu

def collate_func(documents):
    # When pretraining the coverage model, can feed real summaries, or the first K words of the document as summaries (for full unsupervised).
    return [utils_misc.cut300(doc['body']) for doc in documents], [" ".join(doc['body'].split()[:50]) for doc in documents]

dataset = SQLDataset(args.dataset_file)
dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

kw_cov = KeywordCoverage(args.device, keyword_model_file=os.path.join(models_folder, "keyword_extractor.joblib"), n_kws=args.n_kws) # , model_file=os.path.join(models_folder, "news_bert_bs64.bin")
kw_cov.model.train()
print("Loaded model")

param_optimizer = list(kw_cov.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
logplot = LogPlot(os.path.join(logs_folder, "coverage/bert_coverage_%s.log" % (args.experiment)))

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    kw_cov.model, optimizer = amp.initialize(kw_cov.model, optimizer, opt_level="O1") # For now O1. See details at https://nvidia.github.io/apex/amp.html

time_save = time.time()
optim_every = 4

for ib, batch in enumerate(dataloader):
    contents, summaries = batch
    loss, acc = kw_cov.train_batch(contents, summaries)
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    if ib%optim_every == 0:
        optimizer.step()
        optimizer.zero_grad()

    logplot.cache({"loss": loss.item(), "accuracy": acc, "count": len(batch)}, prefix="T_")
    if time.time()-time_save > 60.0:
        logplot.save(printing=True)
        time_save = time.time()
        kw_cov.save_model(os.path.join(models_folder, "bert_coverage_%s.bin" % (args.experiment)))
