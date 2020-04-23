from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForMaskedLM
from torch.nn.modules.loss import CrossEntropyLoss
import torch, os, time, tqdm, numpy as np, h5py

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from collections import Counter

class KeywordExtractor():
    def __init__(self, n_kws=15):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.max_len = 10000
        self.n_kws = n_kws

        self.bert_w2i = {w: i for i, w in enumerate(self.tokenizer.vocab)}
        self.bert_vocab = self.tokenizer.vocab
        # self.dataset = h5py.File("/home/phillab/data/headliner_6M.hdf5")
        # self.dset = self.dataset['name']
        self.keyworder = None
        self.i2w = None
        # self.cache = {}
        # self.cache_keys = []

    def train(self):
        stop_words = ["'", ".", "!", "?", ",", '"', '-', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        stop_indices = set([bert_w2i[w] for w in stop_words if w in bert_w2i])
        dv = DictVectorizer()
        tt = TfidfTransformer()
        self.keyworder = Pipeline([('counter', dv), ('tfidf', tt)])

        def remove_stop_words(tokenized):
            return [w for w in tokenized if w not in stop_indices]

        N = 100000
        text_inputs = [self.tokenizer.encode(dset[i][0].decode()) for i in tqdm.tqdm_notebook(range(N))] # Tokenize
        text_inputs = [remove_stop_words(text) for text in text_inputs] # Remove stop words

        text_inputs = [Counter(text) for text in text_inputs] # Make a Count dictionary
        training_output = self.keyworder.fit_transform(text_inputs)

    def save(self, outfile):
        joblib.dump(self.keyworder, outfile)

    def reload(self, infile):
        self.keyworder = joblib.load(infile)
        self.counter = self.keyworder.named_steps['counter']
        self.i2w = {i:w for w,i in self.counter.vocabulary_.items()}

    def extract_keywords(self, unmasked):
        # if text in self.cache:
        #     return self.cache[text]

        # unmasked = self.tokenizer.encode(text)
        tfidf = self.keyworder.transform([Counter(unmasked)])
        kws = np.argsort(tfidf.toarray()[0])[::-1][:self.n_kws]
        kws_is = [self.i2w[kw] for kw in kws]
        kws_texts = [self.tokenizer.ids_to_tokens[kwi] for kwi in kws_is]
        # print(kws_is, kws_texts)
        outputs = (kws_is, kws_texts)

        # if len(self.cache) > 1000:
        #     del self.cache[self.cache_keys.pop(0)]
        # self.cache[text] = outputs
        # self.cache_keys.append(text)
        return outputs

class KeywordCoverage():
    def __init__(self, device, keyword_model_file, model_file=None, n_kws=15):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.max_len = 10000
        self.vocab_size = self.tokenizer.vocab_size
        self.n_kws = n_kws

        self.mask_id = 103
        self.sep_id  = 102

        self.kw_ex = KeywordExtractor(n_kws=self.n_kws)
        self.kw_ex.reload(keyword_model_file)
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.device = device
        self.model.to(self.device)
        if model_file is not None:
            self.reload_model(model_file)

    def mask_text(self, text_tokenized):
        kws_is, kws_texts = self.kw_ex.extract_keywords(text_tokenized)
        kws_is = set(kws_is)
        # unmasked = self.tokenizer.encode(text)
        masked = [self.mask_id if wi in kws_is else wi for wi in text_tokenized]
        return masked

    def reload_model(self, model_file):
        print(self.model.load_state_dict(torch.load(model_file), strict=False))

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def build_io(self, contents_tokenized, summaries):
        N = len(contents_tokenized)
        maskeds, unmaskeds = [], []
        summ_toks = []
        T1 = time.time()
        for content_tokenized, summary in zip(contents_tokenized, summaries):
            masked = self.mask_text(content_tokenized) # .decode()
            maskeds.append(torch.LongTensor(masked))
            unmaskeds.append(torch.LongTensor(content_tokenized))
            summ_toks.append(torch.LongTensor(self.tokenizer.encode(summary))) # .decode()
        T2 = time.time()
        input_ids = torch.nn.utils.rnn.pad_sequence(maskeds, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(unmaskeds, batch_first=True, padding_value=0)
        input_ids = input_ids[:, :300]
        labels = labels[:, :300]

        summ_toks = torch.nn.utils.rnn.pad_sequence(summ_toks, batch_first=True, padding_value=0)
        summ_toks = summ_toks[:, :100]
        T3 = time.time()
        seps = torch.LongTensor([self.sep_id] * N).unsqueeze(1)
        input_ids = torch.cat((summ_toks, seps, input_ids), dim=1)
        labels = torch.cat((summ_toks, seps, labels), dim=1)
        is_masked = input_ids.eq(torch.LongTensor([self.mask_id])).long()

        # Make the labels classifier friendly
        labels = labels * is_masked + (1-is_masked) * torch.LongTensor([-1])

        T4 = time.time()
        # print(T2-T1, T3-T2, T4-T3)
        labels = labels.to(self.device)
        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)

        return input_ids, is_masked, labels
    def train_batch(self, contents, summaries):
        contents_tokenized = [self.tokenizer.encode(cont) for cont in contents]

        input_ids, is_masked, labels = self.build_io(contents_tokenized, summaries)

        outputs, = self.model(input_ids)
        cross_ent = CrossEntropyLoss(ignore_index=-1)
        loss = cross_ent(outputs.view(-1, self.vocab_size), labels.view(-1))

        num_masks = torch.sum(is_masked, dim=1).float() + 0.1
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=2)
            accs = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / num_masks
        return loss, accs.mean().item()

    def score(self, summaries, contents, bodies_tokenized, lengths=None, extra=None):
        contents_tokenized = bodies_tokenized
        # self.model.eval()
        with torch.no_grad():
            input_ids_w, is_masked_w, labels_w = self.build_io(contents_tokenized, summaries)
            outputs_w, = self.model(input_ids_w)
            preds_w = torch.argmax(outputs_w, dim=2)
            num_masks_w = torch.sum(is_masked_w, dim=1).float() + 0.1
            accs_w = torch.sum(preds_w.eq(labels_w).long() * is_masked_w, dim=1).float() / num_masks_w

            if extra is not None:
                accs_wo = extra # We're in the argmax, and this has already been computed in the sampled
            else:
                input_ids_wo, is_masked_wo, labels_wo = self.build_io(contents_tokenized, [""] * len(contents_tokenized))
                outputs_wo, = self.model(input_ids_wo)
                preds_wo = torch.argmax(outputs_wo, dim=2)
                num_masks_wo = torch.sum(is_masked_wo, dim=1).float() + 0.1
                accs_wo = torch.sum(preds_wo.eq(labels_wo).long() * is_masked_wo, dim=1).float() / num_masks_wo
        score = accs_w - accs_wo
        return score.tolist(), accs_wo

if __name__ == "__main__":
    import utils_tokenizer

    # contents = ["Rowan Williams and Simon Russell Beale: Shakespeare - Spiritual, Secular or Both? [17] Was Shakespeare a secret Catholic in an age of recusansy laws? Or a steadfast Anglican? And what cryptic clues do his plays provide? The Archbishop of Canterbury examines the Bard's relationship with religion. Oxfam Stage, PS5 Gareth Malone: Music for the People [18] Having made choral music cool with kids - sort of - as the beaming maestro in BAFTA-winning BBC series The Choir, Malone now directs his seemingly limitless enthusiasm to the broader classical genre. 5.15pm Ofxam Stage, PS5 Sir Colin Humphreys: Cambridge Series 1: The Mystery of The Last Supper [21] The distinguished physicist turned biblical historian explains the primary revelation of his latest book: that the Last Supper took place on Holy Wednesday, not Maundy Thursday. All down to calendaring, apparently. 5.15pm Llwyfan Cymru - Wales Stage, PS5 Rachel Campbell-Johnston: Mysterious Wisdom [22] From fertile Kent gardens to the pastoral elegance of the Campania countryside, Samuel Palmer was a master of lanscape painting. Rachel Campbell-Johnston discusses her new book on the lynchpin of British Romanticism. 6.30pm Elmley Foundation Theatre, PS5 Anthony Sattin: Lifting the Veil [27] While the UK population's mini-break plans to Egypt may be shelved for the forseeale future, this hasn't dettered Anthony Sattin's infatuation. He traces two centuries of kindred spirits drawn to the beguiling mores of the Land of the Pharoahs. 7.45pm Llwyfan Cymru - Wales Stage, PS5 Simon Mitton - Cambridge Series 3: From Alexandria to Cambridge [29] The secrets of life, the universe and everything have been written in the stars since time began. Astrophysicist and academic Simon Mitton believes they are now more readily available - in books. Here he explores five key works, from Copernicus to Newton. 9.30pm Oxfam Stage, PS8 Jason Byrne: Cirque du Byrne"]
    contents = ["To the chagrin of New York antiques dealers, lawmakers in Albany have voted to outlaw the sale of virtually all items containing more than small amounts of elephant ivory, mammoth ivory or rhinoceros horn. The legislation, which is backed by Gov. Andrew M. Cuomo, will essentially eliminate New York's central role in a well-established, nationwide trade with an estimated annual value of $500 million. Lawmakers say the prohibitions are needed to curtail the slaughter of endangered African elephants and rhinos, which they say is fueled by a global black market in poached ivory, some of which has turned up in New York. The illegal ivory trade has no place in New York State, and we will not stand for individuals who violate the law by supporting it,\" Mr. Cuomo said in a statement on Tuesday, during the debate on the bill. The bill was approved by the Assembly on Thursday, 97 to 2, and passed the Senate, 43 to 17, on Friday morning. Mr. Cuomo is expected to sign it within a week. Assemblyman Robert K. Sweeney, Democrat of Lindenhurst, a sponsor, said that the law \"recognizes the significant impact our state can have on clamping down on illegal ivory sales\" and that it would help rescue elephants from \"ruthless poaching operations run by terrorists and organized crime.\" Dealers and collectors who trade in ivory antiques owned long before the era of mass poaching say the restrictions, which are stiffer than similar federal rules announced in May, will hurt legitimate sellers but do little to protect endangered animals. The real threat to elephants and rhinos, they say, comes from the enormous illicit market in tusks and horns based in China and other Asian nations. \"It is masterful self-deception to think the elephant can be saved by banning ivory in New"]
    summaries = [""]

    models_folder = "/home/phillab/models/"
    model_file = os.path.join(models_folder, "bert_coverage_cnndm_lr4e5_0.bin")
    # model_file = os.path.join(models_folder, "bert_coverage_cnndm_bs64_0.bin")
    kw_cov = KeywordCoverage("cuda", model_file=model_file, keyword_model_file=os.path.join(models_folder, "keyword_extractor.joblib"))
    bert_tokenizer = utils_tokenizer.BERTCacheTokenizer()

    contents_tokenized = [bert_tokenizer.encode(body) for body in contents]
    scores, no_summ_acc = kw_cov.score(summaries, contents, contents_tokenized)

    for body, score, ns_score in zip(contents, scores, no_summ_acc):
        print("----------------")
        print("----------------")
        print("----------------")
        print(body)
        print("---")
        print(score)
        print("---")
        print(ns_score)
