from transformers.optimization import AdamW, WarmupLinearSchedule
from model_generator import GeneTransformer
from torch.utils.data import DataLoader, RandomSampler
from utils_logplot import LogPlot
import torch, os, time, argparse
from datetime import datetime
import utils_hdf5
import getpass, tqdm

# user = os.getlogin()
user = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--dataset_file", type=str, required=True, help="Which dataset file to use.")
parser.add_argument("--task", type=str, help="Which generation task to perform. Can be: cgen (conditionally generate), lm (language modeling) or copy")
parser.add_argument("--max_output_length", required=True, type=int, help="Maximum output length. Saves time if the sequences are short.")

parser.add_argument("--root_folder", type=str, default="/home/"+user+"/")
parser.add_argument("--tokenizer", type=str, default="gpt2", help="Which tokenizer to use: gpt2 or bpecap.")
parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs to run over the data.")
parser.add_argument("--optim_every", type=int, default=4, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--starter_model', default="", help="which model to start with. Leave empty string for random inizialitation")

args = parser.parse_args()

models_folder = os.path.join(args.root_folder, "models/")
logs_folder =   os.path.join(args.root_folder, "logs/")

if args.device == "cuda":
    freer_gpu = str(utils_hdf5.get_freer_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
    args.experiment += "_"+freer_gpu

learning_rate = 2e-5
n_epochs = args.n_epochs

utils_hdf5.DoublePrint("printlog_generator_"+args.experiment+"_"+datetime.now().strftime("%Y-%m-%d")+".log", "a") ## << Wooh

bpe_model = ""
if args.tokenizer == "bpecap":
    bpe_model = os.path.join(models_folder, "m.model")

model = GeneTransformer(tokenizer_type=args.tokenizer, max_output_length=args.max_output_length, device=args.device, bpe_model=bpe_model)
if len(args.starter_model) > 0:
    model.reload(os.path.join(models_folder, args.starter_model))

print("Model loaded")

def collate_func(inps):
    return [inp[0] for inp in inps], [inp[1] for inp in inps]

dataset = utils_hdf5.HDF5Dataset(args.dataset_file, collection_name="name")

N = len(dataset)
N_dev = 500
N_train = N-N_dev
d_train, d_dev = torch.utils.data.dataset.random_split(dataset, [N_train, N_dev])

dl_train = DataLoader(dataset=d_train, batch_size=args.train_batch_size, sampler=RandomSampler(d_train), collate_fn=collate_func)
dl_dev   = DataLoader(dataset=d_dev,   batch_size=20, sampler=RandomSampler(d_dev), collate_fn=collate_func)

# dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

param_optimizer = list(model.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

logplot_file = os.path.join(logs_folder, "generator_"+args.experiment+".log")
summ = LogPlot(logplot_file)

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=n_epochs*len(dl_train))

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model.model, optimizer = amp.initialize(model.model, optimizer, opt_level="O1") # For now O1. See details at https://nvidia.github.io/apex/amp.html

print("Started training")
time_save = time.time()

def map_batch(batch, task):
    sources, targets = batch
    sources = [source.decode() for source in sources]
    targets = [target.decode() for target in targets]

    sources = [s for s in sources]
    if task == "copy":
        targets = sources
    elif task == "cgen":
        targets = [t for t in targets]
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
            scheduler.step()  # Update learning rate schedule
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
