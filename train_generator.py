from transformers.optimization import AdamW
from model_generator import GeneTransformer
from torch.utils.data import DataLoader, RandomSampler
import torch, os, time, argparse, tqdm
from utils_dataset import SQLDataset
from utils_logplot import LogPlot
from datetime import datetime
import utils_misc

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--dataset_file", type=str, required=True, help="Which dataset file to use.")
parser.add_argument("--task", type=str, required=True, help="Which generation task to perform. Can be: `cgen` (conditionally generate),  lm` (language modeling) or `copy`. `cgen` is useful to train a supervised model, when data is available (for example a headline generator, summarizer, etc). `lm` is an unconditional language model, such as the GPT2 model, can be used to train a Fluency model. `copy` can be used to pretrain the generator for the summary_loop, this speeds up training of the summary_loop as the generator already starts with the strong baseline of copying the first K words of the input.")
parser.add_argument("--max_output_length", required=True, type=int, help="Maximum output length. Saves time if the sequences are short.")

parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs to run over the data.")
parser.add_argument("--optim_every", type=int, default=4, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--starter_model', default="", help="which model to start with. Leave empty string for random inizialitation")

args = parser.parse_args()

models_folder = "/home/ubuntu/models/"
logs_folder =   "/home/ubuntu/logs/"

if args.device == "cuda":
    freer_gpu = str(utils_misc.get_freer_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
    args.experiment += "_"+freer_gpu

learning_rate = 2e-5
n_epochs = args.n_epochs

model = GeneTransformer(tokenizer_type="gpt2", max_output_length=args.max_output_length, device=args.device, bpe_model="")
if len(args.starter_model) > 0:
    model.reload(os.path.join(models_folder, args.starter_model))

print("Model loaded")

def collate_func(documents):
    return [utils_misc.cut300(doc['body']) for doc in documents], [doc['title'] for doc in documents]

dataset = SQLDataset(args.dataset_file)

N = len(dataset)
N_dev = 500
N_train = N-N_dev
d_train, d_dev = torch.utils.data.dataset.random_split(dataset, [N_train, N_dev])

dl_train = DataLoader(dataset=d_train, batch_size=args.train_batch_size, sampler=RandomSampler(d_train), collate_fn=collate_func)
dl_dev   = DataLoader(dataset=d_dev,   batch_size=20, sampler=RandomSampler(d_dev), collate_fn=collate_func)

param_optimizer = list(model.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

logplot_file = os.path.join(logs_folder, "generator_%s.log" % (args.experiment))
summ = LogPlot(logplot_file)

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model.model, optimizer = amp.initialize(model.model, optimizer, opt_level="O2") # For now O1. See details at https://nvidia.github.io/apex/amp.html

print("Started training")
time_save = time.time()

def map_batch(batch, task):
    sources, targets = batch

    if task == "cgen":
        pass # already in shape
    elif task == "copy":
        targets = sources
    elif task == "lm":
        targets = sources
        sources = [""] * len(sources)
    return sources, targets

no_preinput = (args.task == "lm")
for _ in range(n_epochs):
    for ib, batch in enumerate(dl_train):
        model.train()
        sources, targets = map_batch(batch, args.task)

        loss = model.train_batch(sources, targets, no_preinput=no_preinput)
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if ib%args.optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        summ.cache({"loss": loss.item(), "count": len(batch)}, prefix="T_")
        if time.time()-time_save > 60.0:

            print("Starting the eval")
            model.eval()

            with torch.no_grad():
                for batch in tqdm.tqdm(dl_dev):
                    sources, targets = map_batch(batch, args.task)
                    loss = model.train_batch(sources, targets, no_preinput=no_preinput)
                    summ.cache({"loss": loss.item(), "count": len(batch)}, prefix="E_")

            summ.save(printing=True)
            time_save = time.time()
            model_output_file = os.path.join(models_folder, "gpt2_"+args.experiment+".bin")
            model.save(model_output_file)
