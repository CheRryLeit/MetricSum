# -*- coding: utf-8 -*-
import sys

sys.path.append('..')
import logging
from logging import handlers
import json
import torch
from rouge import Rouge
import numpy as np

logger = logging.getLogger("Summarization logger")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
th = handlers.TimedRotatingFileHandler(filename='results/info_logs.txt', when='D', encoding='utf-8')
# th.setFormatter('%(asctime)s %(levelname)-8s: %(message)s')
th.formatter = formatter
th.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(th)

# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

class Evaluation(object):
    def __init__(self, opts):
        self.save_dir = opts.save_dir
        self.blocking_win = opts.blocking_win
        self.select_nums = opts.select_nums
        self.opts = opts
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10
        self._hyps = []
        self._refer = []
    def evaluation(self, logits, dataset, blocking=False, bar=0.5):
        for i in range(len(logits)):
            logit = logits[i]
            document, refer, gold_label = dataset.get_sigle_data(i)
            N = len(document)
            if self.select_nums == 0:
                pred_idx = torch.where(logit[0].cpu() > bar)[0]
            else:
                if blocking:
                    pred_idx = self.ngram_blocking(document, logit, self.blocking_win, min(self.select_nums, N))
                else:
                    # print(p_sent.size())
                    topk, pred_idx = torch.topk(logit, min(self.select_nums, N))
                pred_idx = pred_idx.tolist()
            self.correct_num += self.match(pred_idx, gold_label, not_list=True)
            self.predict_num += len(pred_idx)
            self.gold_num += len(gold_label)

            hyps = "\n".join(document[id] for id in pred_idx if id < N)

            self._hyps.append(hyps)
            self._refer.append(refer)
        p, r, f = self._prf()
        rouge = self.rouge()

        return p,r,f, rouge
    def per_eval(self, pred_idx, gold_label):
        self.correct_num += self.match(pred_idx, gold_label)
        self.predict_num += self.count_list(pred_idx)
        self.gold_num += self.count_list(gold_label)
        return self._prf()
    def count_list(self, x):
        count = 0
        for i in x:
            for _ in i:
                count += 1
        return count

    def _prf(self):
        precision = self.correct_num / self.predict_num
        recall = self.correct_num / self.gold_num
        f1_score = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1_score

    def rouge(self):
        reference, candidate = self._refer, self._hyps
        assert len(reference) == len(candidate)
        rouge = Rouge()

        # cand, ref = map(list, zip(*[[' '.join(r), ' '.join(c)] for r, c in zip(reference, candidate)]))
        # scores = rouge.get_scores(cand, ref, avg=True)
        scores = rouge.get_scores(candidate, reference, avg=True)


        recall = [round(scores["rouge-1"]['r'] * 100, 2),
                round(scores["rouge-2"]['r'] * 100, 2),
                round(scores["rouge-l"]['r'] * 100, 2)]
        precision = [round(scores["rouge-1"]['p'] * 100, 2),
                    round(scores["rouge-2"]['p'] * 100, 2),
                    round(scores["rouge-l"]['p'] * 100, 2)]
        f_score = [round(scores["rouge-1"]['f'] * 100, 2),
                round(scores["rouge-2"]['f'] * 100, 2),
                round(scores["rouge-l"]['f'] * 100, 2)]
        print("F_measure: %s Recall: %s Precision: %s\n"
                % (str(f_score), str(recall), str(precision)))

        return f_score[:], recall[:], precision[:]
        

    def match(self, p, l, not_list = False):
        correct = 0
        if not_list:
            correct += len(set(p) & set(l))
        else:
            for i in range(len(l)):
                pp = set(p[i])
                ll = set(l[i])
                correct += len(pp & ll)
            # return correct
        
        return correct
    
            
    def ngram_blocking(self, sents, p_sent, n_win, k):
        """
        
        :param p_sent: [sent_num, 1]
        :param n_win: int, n_win=2,3,4...
        :return: 
        """
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        S = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []
            for i in range(len(pieces) - n_win):
                ngram = " ".join(pieces[i : (i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)
            if overlap_flag == 0:
                S.append(idx)
                ngram_list.extend(sent_ngram)
                if len(S) >= k:
                    break
        S = torch.LongTensor(S)
        # print(sorted_idx, S)
        return S

    def reset(self):
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10


def get_pretrain_word2vec(vocab, opts):
    id2wrod = vocab._id_to_word
    gi2v = np.load(opts.glove_id2vec)
    gw2i = json.load(open(opts.glove_word2id, encoding='utf-8'))
    new_idx2vec = []
    idx_list = []
    for idx, w in id2wrod.items():
        idx_list.append(idx)
        if w in gw2i:
            new_idx2vec.append(gi2v[gw2i[w]])
        else:
            z = gi2v[gw2i['[UNK]']]
            new_idx2vec.append(z)
    return np.stack(new_idx2vec, 0)