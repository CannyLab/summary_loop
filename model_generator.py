from transformers import GPT2LMHeadModel, GPT2Config

import torch.utils.data.dataset
import utils_tokenizer
import torch, tqdm

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class GeneTransformer:
    def __init__(self, max_output_length=25, max_input_length=300, device='cpu', tokenizer_type='gpt2', bpe_model="", starter_model=None):
        if tokenizer_type == "gpt2":
            self.tokenizer = utils_tokenizer.GPT2Tokenizer()
            config = GPT2Config.from_pretrained("gpt2")

        elif tokenizer_type == "bpecap":
            self.tokenizer = utils_tokenizer.BPETokenizer(bpe_model)
            config = GPT2Config.from_dict({"finetuning_task": None, "initializer_range": 0.02,
                                           "layer_norm_epsilon": 1e-05, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "num_labels": 1,
                                           "resid_pdrop": 0.1, "use_bfloat16": False, "vocab_size": self.tokenizer.vocab_size})
        else:
            print("Tokenizer unrecognized. Should be gpt2 or bpecap.")
            exit()

        self.model = GPT2LMHeadModel(config)

        self.model.to(device)
        self.device = device
        if starter_model is not None:
            self.reload(starter_model)

        self.max_output_length = max_output_length
        self.max_input_length = max_input_length

        self.model.train()
        self.mode = "train"

    def reload(self, from_file):
        print(self.model.load_state_dict(torch.load(from_file), strict=False))

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def preprocess_input(self, bodies, special_append=None):
        if special_append is None:
            special_append = [[] for i in range(len(bodies))]
        inputs = [torch.LongTensor(spe+self.tokenizer.encode(body)) for body, spe in zip(bodies, special_append)]
        inputs = pad(inputs, padval=0)
        inputs = inputs[:, :self.max_input_length].to(self.device)
        return inputs

    def preprocess_batch(self, bodies, summaries, special_append=None):
        inputs = self.preprocess_input(bodies, special_append)

        # Big hack
        if special_append is None:
            special_append = [[] for i in range(len(bodies))]

        summaries = [spe+self.tokenizer.encode(summ) for summ, spe in zip(summaries, special_append)]

        summaries = [summ[:(self.max_output_length-1)] for summ in summaries] # We cut short, but we want the end token at the end

        summ_inp = pad([torch.LongTensor([self.tokenizer.start_id]+summ) for summ in summaries], padval=0).to(self.device)
        summ_out = pad([torch.LongTensor(summ+[self.tokenizer.end_id]) for summ in summaries], padval=-1).to(self.device)
        # summ_inp = summ_inp[:, :self.max_output_length].to(self.device)
        # summ_out = summ_out[:, :self.max_output_length].to(self.device)
        return inputs, summ_inp, summ_out

    def train_batch(self, bodies, summaries, special_append=None, no_preinput=False):
        # if self.mode != 'train':
        #     print("BEWARE. Model is not in train mode.")

        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries, special_append)
        past = None
        if not no_preinput:
            _, past = self.model(input_ids=inputs, past_key_values=None)
        logits, _ = self.model(input_ids=summ_inp, past_key_values=past)
        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.contiguous().view(-1))
        return loss

    def train(self):
        self.model.train()
        self.mode = 'train'

    def eval(self):
        self.model.eval()
        self.mode = 'eval'

    def decode_batch(self, bodies, special_append=None, max_output_length=100, sample=False, return_scores=False, return_logprobs=False, input_past=None):
        N = len(bodies)
        current = torch.LongTensor([self.tokenizer.start_id] * N).to(self.device).unsqueeze(1)
        build_up = None
        scores = torch.zeros((N)).to(self.device)
        total_logprobs = []

        # Sometimes, we process the same input, as we run it once as a sampled, and once as an argmax, in which case we should reuse the computation
        if input_past is None:
            inputs = self.preprocess_input(bodies, special_append)
            _, input_past = self.model(input_ids=inputs, past_key_values=None)

        past = input_past
        while build_up is None or (build_up.shape[1] < max_output_length and not all([self.tokenizer.end_id in build for build in build_up])):
            logits, past = self.model(input_ids=current, past_key_values=past)
            probs = torch.nn.functional.softmax(logits, dim=2).squeeze(1)
            logprobs = torch.nn.functional.log_softmax(logits, dim=2)
            if sample:
                current = torch.multinomial(probs, 1)
            else:
                current = torch.argmax(logprobs, dim=2)

            if build_up is None:
                build_up = current
            else:
                build_up = torch.cat((build_up, current), dim=1)

            if return_logprobs:
                selected_logprobs = logprobs[torch.arange(N), 0, current.squeeze()].unsqueeze(1)
                total_logprobs.append(selected_logprobs)

            not_finished = (1-torch.any(build_up ==self.tokenizer.end_id, dim=1).float()).to(self.device)
            scores += not_finished * logprobs[torch.arange(N), :, current.squeeze(1)].squeeze()

        end_id = self.tokenizer.end_id
        build_up = [build.tolist() for build in build_up]
        end_indices = [max_output_length+1 if end_id not in build else build.index(end_id) for build in build_up]
        outputs = [self.tokenizer.decode(build)+"END" for build in build_up]
        outputs = [S[:S.index("END")] for S in outputs]

        if return_logprobs:
            return outputs, torch.cat(total_logprobs, dim=1), build_up, input_past, end_indices
        elif return_scores:
            return outputs, scores.tolist()
        else:
            return outputs

    def decode_beam_batch(self, bodies, beam_size=3, max_output_length=100, sample=False):
        if self.mode != 'eval':
            print("BEWARE. Model is not in eval mode.")
        self.eval() # << Surely you are not training with beam decode?

        batch_size = len(bodies)
        N = batch_size * beam_size
        inputs = self.preprocess_input(bodies)
        next_words = torch.LongTensor([self.tokenizer.start_id] * N).to(self.device).unsqueeze(1)
        build_up = None
        scores = torch.zeros((N)).to(self.device)

        one_every_k = torch.FloatTensor([1] + [0] * (beam_size-1)).repeat(batch_size*beam_size).to(self.device)

        # Sometimes, we process the same input, as we run it once as a sampled, and once as an argmax, in which case we should reuse the computation
        _, input_past = self.model(input_ids=inputs, past_key_values=None)
        input_past = [torch.repeat_interleave(p, repeats=beam_size, dim=1) for p in input_past]

        past = input_past
        while build_up is None or (build_up.shape[1] < max_output_length and not all([self.tokenizer.end_id in build for build in build_up])):
            logits, past = self.model(input_ids=next_words, past_key_values=past)
            probs = torch.nn.functional.softmax(logits, dim=2).squeeze(1)
            logprobs = torch.nn.functional.log_softmax(logits, dim=2)

            if sample:
                all_selects = torch.multinomial(probs, beam_size).unsqueeze(1)
            else:
                _, all_selects = torch.topk(logprobs, k=beam_size, dim=2)

            if build_up is not None:
                not_finished = (1-torch.any(build_up==self.tokenizer.end_id, dim=1).float()).to(self.device)
            else:
                not_finished = torch.ones_like(scores, dtype=torch.float, device=self.device)

            expanded_not_finished = torch.repeat_interleave(not_finished, repeats=beam_size)

            expanded_score = torch.repeat_interleave(scores, repeats=beam_size) # This should be batch_size * beam_size²
            added_score = logprobs[torch.repeat_interleave(torch.arange(N), repeats=beam_size), 0, all_selects.view(-1)]
            expanded_score += (expanded_not_finished*added_score)

            # We don't want you to select from finished beams
            expanded_score -= (1-expanded_not_finished)*(1-one_every_k)*1000.0

            batched_scores = expanded_score.view(batch_size, -1)

            if build_up is None:
                choices = torch.arange(beam_size, device=self.device).repeat(batch_size)
                batched_choices = choices.view(batch_size, beam_size)

            else:
                _, batched_choices = torch.topk(batched_scores, k=beam_size, dim=1) # Going from k² choices per element to k choices.

            batched_tracks = (batched_choices / beam_size).long()
            tracks = beam_size*torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size).to(self.device) + batched_tracks.view(-1)

            selected_scores = batched_scores[torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size), batched_choices.view(-1)]

            # Figure out the kept words to be added to the build-up
            per_batch_selects = all_selects.view(batch_size, -1)
            next_words = per_batch_selects[torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size), batched_choices.view(-1)]
            next_words = next_words.unsqueeze(1)

            # [BOOKKEEPING] Going from k² to k options at each time means we have to swap all the caches around: past, build-up
            if build_up is not None:
                build_up = build_up[tracks, :]
            past = [p[:, tracks, :] for p in past]

            # Update the latest scores, and the current_build
            if build_up is None:
                build_up = next_words
            else:
                build_up = torch.cat((build_up, next_words), dim=1)
            scores = selected_scores.view(-1)

        batched_build_up = build_up.view(batch_size, beam_size, -1)
        batched_scores = scores.view(batch_size, -1)
        # torch.cuda.empty_cache()

        outputs = []
        for beams in batched_build_up:
            out_beams = [self.tokenizer.decode(beam.tolist())+"END" for beam in beams]
            out_beams = [S[:S.index("END")] for S in out_beams]
            outputs.append(out_beams)

        return outputs, batched_scores.tolist()

    def decode(self, bodies, max_output_length=100, max_batch_size=8, beam_size=1, return_scores=False, sample=False, progress=False):
        N = len(bodies)
        outputs = []
        scores = []
        iterator = range(0, N, max_batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            batch_bodies = bodies[i:min(N, i+max_batch_size)]
            with torch.no_grad():
                if beam_size > 1:
                    batch_outputs = self.decode_beam_batch(batch_bodies, beam_size=beam_size, max_output_length=max_output_length, sample=sample)
                else:
                    batch_outputs = self.decode_batch(batch_bodies, max_output_length=max_output_length, sample=sample, return_scores=return_scores)
            if return_scores:
                batch_outputs, batch_scores = batch_outputs
                scores.extend(batch_scores)
            outputs.extend(batch_outputs)

        if return_scores:
            return outputs, scores
        else:
            return outputs

    def score(self, summaries, bodies, bodies_tokenized=None, lengths=None, extra=None):
        # Unconditional rating of the summaries
        self.model.eval()
        # if self.mode != 'eval':
        #     print("BEWARE. Model is not in eval mode.")

        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries)
        summ_out = summ_out.contiguous()

        with torch.no_grad():
            logits, _ = self.model(input_ids=summ_inp, past_key_values=None)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.view(-1)).view(summ_out.shape)
            mask = (summ_inp != torch.LongTensor([0]).to(self.device)).float()
            non_pad_count = torch.sum(mask, dim=1)
            loss_per = torch.sum(loss, dim=1) / non_pad_count

        score = (10.0 - loss_per) / 10.0
        return score.tolist(), None

    def score_pairs(self, bodies, summaries):
        if self.mode != 'eval':
            print("BEWARE. Model is not in eval mode.")

        inputs, summ_inp, summ_out = self.preprocess_batch(bodies, summaries)

        with torch.no_grad():
            _, past = self.model(input_ids=inputs, past_key_values=None)
            logits, _ = self.model(input_ids=summ_inp, past_key_values=past)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(logits.view(-1, self.tokenizer.vocab_size), summ_out.view(-1)).view(summ_out.shape)
            mask = (summ_inp != torch.LongTensor([0]).to(self.device)).float()
            non_pad_count = torch.sum(mask, dim=1)
            loss_per = torch.sum(loss, dim=1) / non_pad_count

        return loss_per.tolist()
