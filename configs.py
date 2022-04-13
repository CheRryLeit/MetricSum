import os
import numpy as np
    
rootPath = '/home/chengyue/extract/RL-EAS'


class Config(object):
    data_path_root = "/home/chengyue/extract/RL-EAS"
    pretrain_root = "/home/chengyue/extract/RL-EAS"
    # pretrain word2vec
    glove_word2id = os.path.join(pretrain_root, "glove/glove_word2id.json")
    glove_id2vec = os.path.join(pretrain_root, "glove/glove_mat.npy")
    # datasets
    cnndm_train = os.path.join(data_path_root, "cnndm/train.label.jsonl")
    cnndm_val = os.path.join(data_path_root, "cnndm/val.label.jsonl")
    cnndm_test = os.path.join(data_path_root, "cnndm/test.label.jsonl")
    # vocab
    vocab_file = "script/cache/CNNDM/vocab"
    vocab_size = 50000

    blocking_win = 3

    embed_train = False

    select_nums = 3
    save_ckpt = '/home/chengyue/extract/RL-EAS/checkpoint/ckpt_gru2.ckpt'

    # Hyperparameters
    seed = 2020
    model_name = "MetricSum"
    sentence_encoder_name = "cnn"
    embedding_dim = 50 #768 230
    num_workers = 0
    batch_size= 128
    lr = 3e-5    # 2e-5
    optim = 'sgd'
    weight_decay = 1e-5
    epochs = 6 
    hidden_size = 50
    blocking = True
    lr_step_size = 2000
    
    cuda = True
    sent_max_len = 100
    doc_max_timesteps = 50
    seed = 2020 
    warmup=True
    warmup_step=300
    ### 是否用bert
    bert = False
    bert_optim = False ##### 用AdamW优化器

    prefix = '-'.join([model_name, sentence_encoder_name, str(hidden_size), str(lr), optim, str(seed)])

    # Eval paramenters
    if not os.path.exists('results'):
        os.mkdir('results')
    save_dir = "results/{}_text.txt".format(prefix)

    # save path 
    load_ckpt = None

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    model_save_path = "checkpoint/{}.pth.tar".format(prefix)

