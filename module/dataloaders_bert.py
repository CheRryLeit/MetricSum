import sys
from transformers import BertTokenizer
sys.path.append('..')
import time
import torch.utils.data as data
import torch.nn.functional as F
import torch
import numpy as np
from .funcs import logger, readJson


class SigDocDataset(data.Dataset):
    def __init__(self, data_path, sent_max_len, doc_max_timesteps):

        self.bert_tokenizer = BertTokenizer.from_pretrained('/home/chengyue/extract/RL-EAS/bert-base-uncased/')
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.ori_data = readJson(data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.ori_data))
        self.size = len(self.ori_data)
    def __getitem__(self, item):
        e = self.ori_data[item]
        e["summary"] = e.setdefault("summary", [])
        text = e["text"]
        summary = e["summary"]  # meiyongdao 
        label = e["label"]
        return self.word2id(text, summary, label)

    def word2id(self, text, summary, label):
        enc_sent_len = []
        enc_sent_input = []

        original_abstract = "\n".join(summary)
        for sent in text:
            article_words = sent.split()

            tokens = self.bert_tokenizer.tokenize(''.join(article_words))
            enc_sent_len.append(len(tokens))  # store the length before padding
            token_id = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            enc_sent_input.append(token_id)  # list of word ids; OOVs are represented by the id for UNK token
        enc_sent_input, mask = self._pad_input(enc_sent_input)
        enc_sent_input = enc_sent_input[:self.doc_max_timesteps]

        return [enc_sent_input, label, mask]


    def get_sigle_data(self, item):
        e = self.ori_data[item]
        e["summary"] = e.setdefault("summary", [])
        text = e["text"]
        summary = e["summary"]
        label = e["label"]
        original_abstract = "\n".join(summary)
        return text, original_abstract, label

    def _pad_input(self, enc_sent_input):
        pad_id = 0
        mask = []
        enc_sent_input_pad = []
        max_len = self.sent_max_len
        mask_matrix = [[1]*len(l) for l in enc_sent_input]
        for i in range(len(enc_sent_input)):
            article_words = enc_sent_input[i].copy()
            single_mask = mask_matrix[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
                single_mask = single_mask[:max_len]

            if len(article_words) < max_len: 
                pading_lens = max_len - len(article_words)
                article_words.extend([pad_id] * pading_lens)
                single_mask = single_mask+ [pad_id] * pading_lens
            enc_sent_input_pad.append(article_words)
            mask.append(single_mask) 
        return enc_sent_input_pad, mask

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m


    def __len__(self):
        return self.size
    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))
        batch = [torch.tensor(d).long() for d in data[0]]
        label = [torch.tensor(l).long() for l in data[1]]
        mask = [torch.tensor(l).long() for l in data[2]]
        return batch, label, mask

def get_dataloader(path, vocab=None, opts=None, shuffle=False):
    dataset = SigDocDataset(path, opts.sent_max_len, opts.doc_max_timesteps)
    loader = data.DataLoader(dataset, opts.batch_size, shuffle=shuffle, num_workers=opts.num_workers, collate_fn=dataset.collate_fn)
    return loader

if __name__ == '__main__':
    

                # data_path, sent_max_len, doc_max_timesteps
    d = SigDocDataset("/home/shiyanshi/chengyue/extract/RL-EAS/cnndm/val.label.jsonl", doc_max_timesteps=50, sent_max_len=100)
    print(d[2])
    print("")

