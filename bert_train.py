import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from detect.bert_trainer import BertTrainer as Trainer
from detect.bert_dataset import TextClassificationDataset, TextClassificationCollator
from detect.utils import read_text

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='distilbert-base-uncased')
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_false')
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer, valid_ratio=.2):
    targets, contexts, responses = read_text(fn)
    print(targets[:5])
    unique_targets = list(set(targets))
    target_to_index = {}
    index_to_target = {}
    for i, target in enumerate(unique_targets):
        target_to_index[target] = i
        index_to_target[i] = target
    print(target_to_index)
    targets = list(map(target_to_index.get, targets))
    print(targets[:5])
    shuffled = list(zip(contexts, responses, targets))
    random.shuffle(shuffled)
    contexts = [e[0] for e in shuffled]
    responses = [e[1] for e in shuffled]
    targets = [e[2] for e in shuffled]
    idx = int(len(contexts) * (1 - valid_ratio))

    train_loader = DataLoader(
        TextClassificationDataset(contexts[:idx], responses[:idx], targets[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(contexts[idx:], responses[idx:], targets[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_target

def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer

def main(config):
    # Get pretrained tokenizer.
    tokenizer = DistilBertTokenizer.from_pretrained(config.pretrained_model_name, strip_accents=False)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_target = get_loaders(
        config.train_fn,
        tokenizer,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model = DistilBertForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_target)
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_target,
        'tokenizer': tokenizer,
    }, config.model_fn)



if __name__ == '__main__':
    config = define_argparser()
    main(config)
