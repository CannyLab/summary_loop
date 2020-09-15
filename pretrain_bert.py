from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertForPreTraining
from torch.utils.data import DataLoader, RandomSampler
import torch, os, time, utils_misc, argparse
from utils_dataset import SQLDataset
from utils_logplot import LogPlot
import random

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_nb", type=int, default=3, help="Which GPU to use. For now single GPU.")
parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
parser.add_argument("--optim_every", type=int, default=8, help="Optimize every x backprops. A multiplier to the true batch size.")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--dataset_file", type=str, default="/home/phillab/data/headliner_6M.hdf5", help="Which dataset file to use.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(args.gpu_nb)

learning_rate = 2e-5
n_epochs = 3

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.max_len = 10000
model = BertForPreTraining.from_pretrained("bert-base-uncased")
model.to(args.device)
print("Model loaded")

vocab_size = tokenizer.vocab_size

summ = LogPlot("/home/phillab/logs/bert-base-uncased/bert_news.log")

def random_word(tokens, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            # -> rest 10% randomly keep current token
            output_label.append(tokenizer.vocab[token])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_example_to_features(tokens_a, tokens_b, max_seq_length, tokenizer):
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #

    tokens =      ["[CLS]"] + tokens_a +             ["[SEP]"] + tokens_b +              ["[SEP]"]
    segment_ids = [0] +      (len(tokens_a) * [0]) + [0] +       (len(tokens_b) * [1]) + [1] 
    lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    pad_amount = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_amount
    input_ids += [0] * pad_amount
    segment_ids += [0] * pad_amount
    lm_label_ids += [-1] * pad_amount

    return input_ids, input_mask, segment_ids, lm_label_ids

def collate_func(inps):
    bodies = [inp['body'] for inp in inps]
    bodies_tokenized = [tokenizer.tokenize(body) for body in bodies]

    max_length = 400
    half_length = int(max_length/2)

    is_next_labels = []
    mid_point = int(len(inps)/2)
    batch_ids, batch_mask, batch_segments, batch_lm_label_ids, batch_is_next = [], [], [], [], []
    for i in range(mid_point):
        is_next = 1 if random.random() < 0.5 else 0

        tokens_a = bodies_tokenized[i]
        if is_next == 0:
            tokens_b = bodies_tokenized[i]
        else:
            tokens_b = bodies_tokenized[i+mid_point]
        tokens_a = tokens_a[:half_length]
        tokens_b = tokens_b[half_length:max_length]
        input_ids, input_mask, segment_ids, lm_label_ids = convert_example_to_features(tokens_a, tokens_b, max_length, tokenizer)

        batch_ids.append(input_ids)
        batch_mask.append(input_mask)
        batch_segments.append(segment_ids)
        batch_lm_label_ids.append(lm_label_ids)
        batch_is_next.append(is_next)

    batch_ids = torch.LongTensor(batch_ids)
    batch_mask = torch.LongTensor(batch_mask)
    batch_segments = torch.LongTensor(batch_segments)
    batch_lm_label_ids = torch.LongTensor(batch_lm_label_ids)
    batch_is_next = torch.LongTensor(batch_is_next)

    return batch_ids, batch_mask, batch_segments, batch_lm_label_ids, batch_is_next

dataset = SQLDataset(args.dataset_file)
dataloader = DataLoader(dataset=dataset, batch_size=2*args.train_batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

model.train()

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=n_epochs*len(dataloader))

time_save = time.time()

for _ in range(n_epochs):
    for ib, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch

        loss, mlm_logits, is_next_logits = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)

        loss.backward()
        is_next_acc = is_next.eq(torch.argmax(is_next_logits, dim=1)).float().mean().item()


        num_predicts = (1.0 - lm_label_ids.eq(-1)).sum().item()
        mlm_acc = (lm_label_ids.view(-1).eq(torch.argmax(mlm_logits,dim=2).view(-1)).float().sum()/num_predicts).item()

        if ib%args.optim_every == 0:
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        summ.cache({"loss": loss.item(), "mlm_acc": mlm_acc, "is_next_acc": is_next_acc}, prefix="T_")
        if time.time()-time_save > 60.0:
            summ.save(printing=True)
            time_save = time.time()
            torch.save(model.state_dict(), "/home/phillab/models/news_bert_bs"+str(args.optim_every*args.train_batch_size)+".bin")
