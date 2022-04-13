import sys

sys.path.append('..')
import time
import torch.utils.data as data
import torch.nn.functional as F
import torch
import numpy as np
from .funcs import logger, readJson


class SigDocDataset(data.Dataset):
    def __init__(self, data_path, vocab, sent_max_len, doc_max_timesteps):
        self.vocab = vocab
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
        summary = e["summary"]
        label = e["label"]
        return self.text2idx(text, summary, label)

    def text2idx(self, text, summary, label):
        enc_sent_len = []
        enc_sent_input = []

        original_abstract = "\n".join(summary)
        for sent in text:
            article_words = sent.split()
            enc_sent_len.append(len(article_words))  # store the length before padding
            enc_sent_input.append([self.vocab.word2id(w.lower()) for w in
                                        article_words])  # list of word ids; OOVs are represented by the id for UNK token
        enc_sent_input = self._pad_input(enc_sent_input)
        # label_shape = (len(text), len(label))
        # label_matrix = np.zeros(label_shape, dtype=int)
        # if label != []:
        #     label_matrix[np.array(label), np.arange(len(label))] = 1
        enc_sent_input = enc_sent_input[:self.doc_max_timesteps]
        # label = self.pad_label_m(label_matrix)
        return [enc_sent_input, label]

    def get_sigle_data(self, item):
        e = self.ori_data[item]
        e["summary"] = e.setdefault("summary", [])
        text = e["text"]
        summary = e["summary"]
        label = e["label"]
        original_abstract = "\n".join(summary)
        return text, original_abstract, label

    def _pad_input(self, enc_sent_input):
        pad_id = self.vocab.word2id('[PAD]')
        enc_sent_input_pad = []
        max_len = self.sent_max_len
        for i in range(len(enc_sent_input)):
            article_words = enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            enc_sent_input_pad.append(article_words)
        return enc_sent_input_pad

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
        label = data[1]
        label = [torch.tensor(l).long() for l in data[1]]
        return batch, label

def get_dataloader(path, vocab, opts, shuffle=False):
    if opts.bert:

        pass
    else:
        dataset = SigDocDataset(path, vocab, opts.sent_max_len, opts.doc_max_timesteps)
        loader = data.DataLoader(dataset, opts.batch_size, shuffle=shuffle, num_workers=opts.num_workers, collate_fn=dataset.collate_fn)
    return loader

if __name__ == '__main__':
    from vocabulary import Vocab
    vocab = Vocab(r"D:\Project\NLG\MetricLearningDemo\script\cache\CNNDM\vocab", 50000)
    d = SigDocDataset(r"D:\Project\NLG\DATA\datasets\cnndm\val.label.jsonl", vocab, doc_max_timesteps=50, sent_max_len=100)
    print(d[2])
    print("")

