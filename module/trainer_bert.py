import os
import numpy as np
import sys
import time
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from .funcs import logger
from transformers import AdamW, get_linear_schedule_with_warmup
from .funcs import Evaluation
def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class SumTrainer(object):

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, opts):

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.opts = opts
        self.data_eval = Evaluation(opts)

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def train(self, model):
        print("Start training...")
        logger.info("[INFO] Start training... {}.".format(self.opts.prefix))
        # Init
        if self.opts.bert_optim:
            print('Use bert optim!')
            # assert 0
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            #### 总的训练步数
            train_iter = (len(self.train_data_loader.dataset) / self.opts.batch_size)* self.opts.epochs
            optimizer = AdamW(parameters_to_optimize, lr=self.opts.lr, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*train_iter,
                                                        num_training_steps=train_iter)
        else:
            if self.opts.optim == "sgd":
                pytorch_optim = optim.SGD
            elif self.opts.optim == "adam":
                pytorch_optim = optim.Adam
            optimizer = pytorch_optim(model.parameters(),self.opts.lr, weight_decay=self.opts.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.lr_step_size)

            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
            #                                             num_training_steps=train_iter)

        if self.opts.load_ckpt:
            checkpoint = self.__load_model__(self.opts.load_ckpt)
            state_dict = checkpoint['state_dict']
            scheduler_sd = checkpoint['scheduler_sd']
            optimizer_sd = checkpoint['optimizer_sd']
            best_rouge = checkpoint['best_rouge']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            optimizer.load_state_dict(optimizer_sd)
            scheduler.load_state_dict(scheduler_sd)
        else:
            best_rouge = 0

        model.train()

        # Training
        not_best_count = 0  # Stop training after several epochs without improvement.
        glob_steps = 0
        # logger.info("start tarining {}.".format())
        for epoch in range(self.opts.epochs):
            avg_loss = 0.0
            self.data_eval.reset()
            start = time.time()
            logger.info("[INFO] LR is {} on {} epochs training.".format(optimizer.param_groups[0]['lr'], epoch))
            for i, data in enumerate(self.train_data_loader):
                document  = data[0]
                label = data[1]
                mask = data[2]
                if self.opts.cuda:
                    for k in range(len(document)):
                        document[k] = document[k].cuda()
                        label[k] = label[k].cuda()
                        mask[k] = mask[k].cuda()
                #### forword
                logits = model(document, mask)
                # model.forward(docs)
                loss = model.loss(logits, label)
                pred = []
                for p in logits:
                    idx = np.where(p.cpu()>0.5)[0]
                    idx = idx.tolist()
                    pred.append(idx)
                p, r, f = self.data_eval.per_eval(pred, label)
                loss.backward()  # retain_graph=True
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss += loss.data.item()
                sys.stdout.write(
                    'epoch:{0:4} | step: {1:4} | loss: {2:2.6f}, p: {3:3.2f}%, r: {4:3.2f}%, f1-score: {5:3.2f}%'
                    .format(epoch, i + 1, avg_loss / (i+1), p, r, f) + '\r')
                sys.stdout.flush()
                glob_steps += 1
                # break
                if glob_steps*self.opts.batch_size > 20000:
                    logger.info("[INFO] Per 20000 Eval val dataets!")
                    precision, recall, f1_score, rouge = self.eval(model, self.val_data_loader)
                    model.train()
                    rouge = rouge[0][0]
                    if rouge > best_rouge:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict(), 'optimizer_sd':optimizer.state_dict(), \
                        'scheduler_sd':scheduler.state_dict(), 'best_rouge':rouge}, self.opts.save_ckpt)
                        best_rouge = rouge
            
            logger.info("[INFO] Finish %d epochs training. Total time is %f, F1 is %f, loss is %f", epoch,
                    time.time() - start, f, avg_loss / (i+1))
            logger.info("[INFO] Eval val dataets!")
            precision, recall, f1_score, rouge = self.eval(model, self.val_data_loader)
            model.train()
            rouge = rouge[0][0]
            if rouge > best_rouge:
                print('Best checkpoint')
                torch.save({'state_dict': model.state_dict(), 'optimizer_sd':optimizer.state_dict(), \
                'scheduler_sd':scheduler.state_dict(), 'best_rouge':rouge}, self.opts.save_ckpt)
                best_rouge = rouge
        
        print("\n####################\n")
        print("Finish training "+self.opts.model_name)

    def eval(self, model, data_loader):
        print("")
        model.eval()
        all_logits = []
        with torch.no_grad():
            start = time.time()
            for it, data in enumerate(data_loader):
                document  = data[0]
                label = data[1]
                mask = data[2]
                if self.opts.cuda:
                    for k in range(len(document)):
                        document[k] = document[k].cuda()
                        label[k] = label[k].cuda()
                        mask[k] = mask[k].cuda()
                logits = model(document, mask)

                pred = []
                for p in logits:
                    idx = np.where(p.cpu()>0.5)[0]
                    idx = idx.tolist()
                    pred.append(idx)
                p, r, f = self.data_eval.per_eval(pred, label)
                for i in range(len(logits)):
                    all_logits.append(logits[i])
                # break

                sys.stdout.write('[EVAL] step: {0:4} | f-score: {1:3.2f}%'.format(it + 1, f) + '\r')
                sys.stdout.flush()
            print("")
        self.data_eval.reset()
        logger.info("[INFO] Finish eval! Total time is %f, F1 is %f", time.time() - start, f)
        start = time.time()
        precision, recall, f1_score, r = self.data_eval.evaluation(all_logits, data_loader.dataset, blocking=self.opts.blocking)
        logger.info("[INFO] Finish eval rouge! Total time is %f, Rouge(R-1, R-2, R-L) is %s", time.time() - start, str(r[0])) #rouge1
        return precision, recall, f1_score, r
