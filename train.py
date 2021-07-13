import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from detect.trainer import Trainer
from detect.dataset import DataLoader

from detect.models.rnn import RNNClassifier

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)
    
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--use_batch_norm', action='store_true')


    config = p.parse_args()

    return config

def main(config):
    loaders = DataLoader(
        train_fn = config.train_fn,
        batch_size = config.batch_size,
        min_freq = config.min_vocab_freq,
        max_vocab = config.max_vocab_size,
        device = config.gpu_id
    )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )

    con_vocab_size = len(loaders.context.vocab)
    res_vocab_size = len(loaders.response.vocab)
    n_classes = len(loaders.target.vocab)
    print('|con_vocab| =', con_vocab_size,
        '|res_vocab| =', res_vocab_size,
        '|classes| =', n_classes)

    if config.rnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        model = RNNClassifier(
            input_size=res_vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )



    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'config': config,
        'vocab': loaders.response.vocab,
        'classes': loaders.target.vocab,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
