from transformers import GPT2Tokenizer as GPT2Tok
from transformers import BertTokenizer as BertTok
import sentencepiece as spm

class Capita:
    def forward(self, text):
        # words = nltk.tokenize.word_tokenize(text)
        words = text.split(" ")
        final_words = []
        for word in words:
            if not word.isalpha():
                final_words.append(word.lower())
            else:
                if word.islower():
                    pass
                elif word.isupper():
                    final_words.append("⇧")
                elif word[0].isupper() and word[1:].islower():
                    final_words.append("↑")
                else:
                    final_words.append("↑")
                final_words.append(word.lower())
        return " ".join(final_words)

    def backward(self, text):
        words = text.split(" ")
        final_words = []
        all_caps = False
        capitalized = False
        for w in words:
            if w == "⇧":
                all_caps = True
            elif w == "↑":
                capitalized = True
            else:
                final_word = w
                if all_caps:
                    final_word = final_word.upper()
                elif capitalized:
                    if len(final_word) <= 1:
                        final_word = final_word.upper()
                    else:
                        final_word = final_word[0].upper()+final_word[1:]
                final_words.append(final_word)
                all_caps = False
                capitalized = False
        return " ".join(final_words)

class BPETokenizer:
    def __init__(self, bpe_model, use_capita=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(bpe_model)
        self.use_capita = use_capita

        self.pad_tok, self.start_tok, self.end_tok = "<pad>", "<start>", "<end>"
        self.pad_id, self.start_id, self.end_id = tuple(self.sp.piece_to_id(p) for p in [self.pad_tok, self.start_tok, self.end_tok])

        self.vocab_size = self.sp.get_piece_size()

        if self.use_capita:
            self.cpt = Capita()

    def tokenize(self, text):
        if len(text) == 0:
            return []
        if text[:len(self.start_tok)] == self.start_tok and text[len(self.start_tok)] != " ":
            text = text.replace(self.start_tok, self.start_tok+" ")

        if self.use_capita:
            text = self.cpt.forward(text)
        tokens = self.sp.encode_as_pieces(text)
        tokens = [w for i, w in enumerate(tokens) if (i < (len(tokens)-1) and tokens[i+1] not in ["⇧", "↑"]) or i==(len(tokens)-1)]
        if tokens[0] == "▁":
            tokens = tokens[1:]
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        token_ids = [self.sp.piece_to_id(w) for w in tokens]
        return token_ids

    def decode(self, token_ids):
        text = self.sp.decode_ids(token_ids).replace("⇧", " ⇧").replace("↑", " ↑")
        if self.use_capita:
            text = self.cpt.backward(text)
        text = text.replace(self.start_tok+" ", self.start_tok)
        return text

class BERTCacheTokenizer:
    def __init__(self):
        self.cache = {}
        self.cache_keys = []
        self.tokenizer = BertTok.from_pretrained("bert-base-uncased")
        # self.tokenizer.max_len = 10000 # This was removed in later transformer tokenizers

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]

        output = self.tokenizer.encode(text)

        if len(self.cache) > 1000:
            del self.cache[self.cache_keys.pop(0)]
        self.cache[text] = output
        self.cache_keys.append(text)
        return output

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tok.from_pretrained("gpt2")
        # self.tokenizer.max_len = 10000

        self.pad_tok, self.start_tok, self.end_tok = "<PAD>", " ST", " END"

        self.pad_id = 0
        self.start_id = self.tokenizer.encode(self.start_tok)[0]
        self.end_id = self.tokenizer.encode(self.end_tok)[0]
        self.vocab_size = self.tokenizer.vocab_size

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
