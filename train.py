import torch
from configs import Config
from module.vocabulary import Vocab
from module.funcs import get_pretrain_word2vec
import random
import numpy as np
def seed_torch(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
if __name__ == "__main__":
    opts = Config()
    seed_torch(opts.seed)
    vocab = Vocab(opts.vocab_file, opts.vocab_size)
     
    w2vec = get_pretrain_word2vec(vocab, opts)
    if opts.bert:
        from models.simple_sum_cert import MetricSum
        from module.trainer_bert import SumTrainer
        from module.dataloaders_bert import get_dataloader
    else:
        from models.simple_sum_gru import MetricSum
        # from models.simple_sum import MetricSum
        # from models.metric_cnn import MetricSum
        from module.trainer import SumTrainer
        from module.dataloaders import get_dataloader

    train = get_dataloader(opts.cnndm_train, vocab, opts, shuffle=True)
    val = get_dataloader(opts.cnndm_val, vocab, opts)
    test = get_dataloader(opts.cnndm_test, vocab, opts)
    

    # model = MetricSum(opts, word2vec=w2vec)
    model = MetricSum(opts)
    if opts.cuda: torch.cuda.set_device(0)
    if opts.cuda: model.cuda()
    
    trainer = SumTrainer(train, val, test, opts)
    if True:
        trainer.train(model)
    else:
        # save_ckpt ="checkpoint/MetricSum-cnn-230-0.0002-adam-2020.pth.tar"
        save_ckpt = 'checkpoint/ckpt_gru.ckpt'  
        checkpoint = trainer.__load_model__(save_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        print('test model on test data!')
        print('This best model F1-score(R-1) is {} on val datasets...'.format(checkpoint['best_rouge']))
        trainer.eval(model, test)
