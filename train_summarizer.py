from transformers.optimization import AdamW, WarmupLinearSchedule
from model_generator import GeneTransformer
from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime, timedelta
from utils_logplot import LogPlot
import torch, os, sys, time, argparse, numpy as np
import utils_hdf5, utils_tokenizer

from coverage import KeywordCoverage
from fluency import FluencyCoLA, PatternPenalty, LengthPenalty, RepeatPenalty
import threading, queue
import torch.utils.data.dataset

user = os.getlogin()

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--dataset_file", type=str, required=True, help="Which dataset file to use. Can be full path or the root folder will be attached.")

parser.add_argument("--root_folder", type=str, default="/home/"+user+"/")
parser.add_argument("--train_batch_size", type=int, default=5, help="Training batch size.")
parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs to run over the data.")
parser.add_argument("--optim_every", type=int, default=4, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--max_output_length", type=int, default=25, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--save_every", type=int, default=60, help="Number of seconds between any two saves.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--model_start", type=str, default="models/headliner/gpt2_headliner_newsgpt2.bin", help="What should the model file start with.")
parser.add_argument("--log_folder", type=str, default="", help="What should the model file start with.")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--mini_dataset', type=int, default=-1, help="If I should use a smaller sized dataset. -1 disables this, entire dataset is used.")
parser.add_argument("--ckpt_every", type=int, default=600, help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.")
parser.add_argument("--ckpt_lookback", type=int, default=300, help="When checkpointing, will consider the avg total score of the last x samples.")

args = parser.parse_args()
if args.device == "cuda":
    freer_gpu = str(utils_hdf5.get_freer_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)
    args.experiment += "_"+freer_gpu

models_folder = os.path.join(args.root_folder, "models/")
args.model_start = os.path.join(models_folder, args.model_start)
args.log_folder = os.path.join(args.root_folder, "logs/", args.log_folder)
args.dataset_file = os.path.join(args.root_folder, args.dataset_file)

ckpt_every = args.ckpt_every
ckpt_lookback = int((args.ckpt_lookback+args.train_batch_size-1)/args.train_batch_size)
total_score_history = []
best_ckpt_score = None
ckpt_file = os.path.join(models_folder, "summarizer_"+args.experiment+"_ckpt.bin")
ckpt_optimizer_file = os.path.join(models_folder, "summarizer_optimizer_"+args.experiment+"_ckpt.bin")

learning_rate = 2e-5
n_epochs = args.n_epochs
utils_hdf5.DoublePrint("printlog_summarizer_"+args.experiment+"_"+datetime.now().strftime("%Y-%m-%d")+".log", "a") ## << Wooh

if args.device == "cuda":
    print("Training on GPU "+str(freer_gpu))

bert_tokenizer = utils_tokenizer.BERTCacheTokenizer()
print("---------------")

summarizer = GeneTransformer(max_output_length=args.max_output_length, device=args.device, tokenizer_type='gpt2', starter_model=args.model_start)
print("Summarizer loaded")

def collate_func(inps):
    return [inp[0].decode() for inp in inps], [inp[1].decode() for inp in inps]

param_optimizer = list(summarizer.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

logplot_file = os.path.join(args.log_folder, "gpt2_unsumm_"+args.experiment+".log")
logplot = LogPlot(logplot_file)

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10000, t_total=500000)
time_save = time.time()
time_ckpt = time.time()

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    summarizer.model, optimizer = amp.initialize(summarizer.model, optimizer, opt_level="O1") # For now O1. See details at https://nvidia.github.io/apex/amp.html

print("Loading scorers")

scorers = [{"name": "coverage", "importance": 10.0, "sign": 1.0, "model": KeywordCoverage(args.device, keyword_model_file=os.path.join(models_folder, "keyword_extractor.joblib"), model_file=os.path.join(models_folder, "bert_coverage_google_cnndm_length15_1.bin"))},
           {"name": "fluency", "importance": 2.0, "sign": 1.0, "model": GeneTransformer(max_output_length=args.max_output_length, device=args.device, starter_model=os.path.join(models_folder, "news_gpt2_bs32.bin"))},
           # {"name": "fluency", "importance": 2.0, "sign": 1.0, "model": FluencyCoLA(args.device,     model_file=os.path.join(models_folder, "bert_fluency_cola.bin"))},
           {"name": "patpen", "importance": 5.0, "sign": -1.0, "model": PatternPenalty()},
           {"name": "lengthpen", "importance": 2.0, "sign": -1.0, "model": LengthPenalty(args.max_output_length)},
           {"name": "reppen", "importance": 2.0, "sign": -1.0, "model": RepeatPenalty()}
           ]

def background_tokenizer(bodies, out_queue):
    out_queue.put([bert_tokenizer.encode(body) for body in bodies])

my_queue = queue.Queue()

print("Started training")

all_dataset = utils_hdf5.HDF5Dataset(args.dataset_file, collection_name="name")
is_mini_dataset = args.mini_dataset > 0
while True:
    print("STARTING NEW MINI DATASET")
    if is_mini_dataset:
        N = len(all_dataset)
        dataset, leftover = torch.utils.data.dataset.random_split(all_dataset, [args.mini_dataset, N-args.mini_dataset])
        N = len(dataset)
    else:
        dataset = all_dataset

    print("Dataset size:", len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

    for epi in range(n_epochs):
        print("=================== EPOCH",epi, "===================")
        for ib, batch in enumerate(dataloader):
            Timer = {}

            T1 = time.time()
            log_obj = {}

            bodies, gold_summaries = batch
            bodies = [" ".join(body.split(" ")[:300]) for body in bodies]

            # We run tokenization in the background, as it is BERT tokenization only used after the summarizer has run. Saves about 5% of time.
            thread1 = threading.Thread(target = background_tokenizer, args = (bodies, my_queue))
            # bodies_bert_tokenized = [bert_tokenizer.enncode(body) for body in bodies] # This is the not background version
            thread1.start()

            T2 = time.time()
            Timer["preprocessing_starting"] = T2-T1

            # T1b = time.time()
            sampled_summaries, sampled_logprobs, sampled_tokens, input_past, sampled_end_idxs = summarizer.decode_batch(bodies, max_output_length=args.max_output_length, return_logprobs=True, sample=True)

            T3 = time.time()
            Timer["generator_sampled"] = T3-T2
            with torch.no_grad():
                argmax_summaries, argmax_end_idxs = summarizer.decode_batch(bodies, max_output_length=args.max_output_length, input_past=input_past)
            T4 = time.time()
            Timer["generator_argmax"] = T4-T3

            selected_logprobs = torch.sum(sampled_logprobs, dim=1)
            batch_size, seq_length = sampled_logprobs.shape

            # We join it here, saying the tokenization that's been running in the background should be done by now.
            thread1.join()
            bodies_bert_tokenized = my_queue.get()

            scores_track = {}
            total_sampled_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
            total_argmax_scores = torch.FloatTensor([0.0] * batch_size).to(args.device)
            for scorer in scorers:
                T = time.time()
                sampled_scores, extra = scorer['model'].score(sampled_summaries, bodies, bodies_tokenized=bodies_bert_tokenized, extra=None, lengths=sampled_end_idxs)
                sampled_scores = torch.FloatTensor(sampled_scores).to(args.device)

                argmax_scores, _ = scorer['model'].score(argmax_summaries, bodies, bodies_tokenized=bodies_bert_tokenized, extra=extra, lengths=argmax_end_idxs)
                argmax_scores  = torch.FloatTensor(argmax_scores).to(args.device)

                Timer["scores_"+scorer['name']] = time.time()-T
                total_sampled_scores += (scorer['sign'])*(scorer['importance'])*sampled_scores
                total_argmax_scores  += (scorer['sign'])*(scorer['importance'])*argmax_scores
                log_obj[scorer['name']+"_score"] = sampled_scores.mean().item()
                scores_track[scorer['name']+"_scores"] = sampled_scores

            T5 = time.time()
            Timer['all_scores'] = T5-T4
            Loss = torch.mean((total_argmax_scores - total_sampled_scores) * selected_logprobs)

            if args.fp16:
                with amp.scale_loss(Loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                Loss.backward()

            T6 = time.time()
            Timer['backward'] = T6-T5

            if ib%args.optim_every == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            T7 = time.time()
            Timer['optim'] = T7-T6

            # log_obj['summary_nwords'] = int(np.mean([summ.count(" ")+1 for summ in sampled_summaries]))
            avg_total = total_sampled_scores.mean().item()

            total_score_history.append(avg_total)
            log_obj['summary_nwords'] = int(np.mean(sampled_end_idxs))
            log_obj['loss'] = Loss.item()
            log_obj['total_score'] = avg_total
            log_obj['count'] = batch_size
            logplot.cache(log_obj, prefix="T_")

            Tfinal = time.time()
            Timer['total'] = Tfinal - T1
            # print(Timer)

            if not is_mini_dataset and (time.time()-time_save > args.save_every):
                print("==========================================")
                print(bodies[0])
                print("-----------")
                print(sampled_summaries[0])
                print("-----------")
                print("Total score:", total_sampled_scores[0].item())
                for scorer in scorers:
                    print(scorer['name']+" score:", scores_track[scorer['name']+"_scores"][0].item())
                print("-----------")

                logplot.save(printing=True)
                # print(Timer)

                time_save = time.time()
                print("==========================================")

            if ckpt_every > 0 and len(total_score_history) > ckpt_lookback:
                current_score = np.mean(total_score_history[-ckpt_lookback:])
                
                if time.time()-time_ckpt > ckpt_every:
                    revert_ckpt = best_ckpt_score is not None and current_score < min(1.2*best_ckpt_score, 0.8*best_ckpt_score) # Could be negative or positive
                    print("================================== CKPT TIME, "+str(datetime.now())+" =================================")
                    print("Previous best:", best_ckpt_score)
                    print("Current Score:", current_score)
                    print("[CKPT] Am I reverting?", ("yes" if revert_ckpt else "no! BEST CKPT"))
                    if revert_ckpt:
                        summarizer.model.load_state_dict(torch.load(ckpt_file))
                        optimizer.load_state_dict(torch.load(ckpt_optimizer_file))
                    time_ckpt = time.time()
                    print("==============================================================================")
        
                if best_ckpt_score is None or current_score > best_ckpt_score:
                    print("[CKPT] Saved new best at:", current_score, "["+str(datetime.now())+"]")
                    best_ckpt_score = current_score
                    torch.save(summarizer.model.state_dict(), ckpt_file)
                    torch.save(optimizer.state_dict(), ckpt_optimizer_file)

        if is_mini_dataset:
            logplot.save(printing=True)
