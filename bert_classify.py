import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=512)
    
#    p.add_argument('--drop_rnn', action='store_true')
#    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config

def read_text(max_length=512):
    contexts, responses = [], []

    for line in sys.stdin:
        if line == "\n":
            break
        if line.strip() != '':
            print(line)
            text = line.strip().split('\t')
            print(text)
            context = text[0]
            response = text[1]
            contexts += [context]
            responses += [response]
    return contexts, responses

def make_text(tokenizer, context, response):
    con = tokenizer(context, max_length=128, truncation=True, return_tensors='pt')
    res = tokenizer(response, max_length=512, truncation=True, return_tensors='pt')
    text = torch.cat([con['input_ids'], res['input_ids']], dim=-1)
    attention_mask = torch.cat([con['attention_mask'], res['attention_mask']], dim=-1)
    pad_length = 512 - text.size(1)
    if pad_length > 0:
        zero_pad = torch.tensor([0]*pad_length).unsqueeze(0)
        text = torch.cat([text, zero_pad], dim=-1)
        attention_mask = torch.cat([attention_mask, zero_pad], dim=-1)
    
    text = text[:, :512]
    attention_mask = attention_mask[:, :512]
    return text, attention_mask

def main(config):
    
    saved_data = torch.load(
        config.model_fn,
        map_location = 'cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    classes = saved_data['classes']
    print(classes)

    contexts, responses = read_text()

    with torch.no_grad():
        tokenizer = DistilBertTokenizer.from_pretrained(train_config.pretrained_model_name, strip_accents=False)

        model = DistilBertForSequenceClassification.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(classes)
        )
        model.load_state_dict(bert_best)
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        model.eval()
        y_hats = []

        for idx in range(0, len(responses), config.batch_size):
            ts, ats = [], []
            for con, res in zip(contexts[idx:idx+config.batch_size], responses[idx:idx+config.batch_size]):
                t, a = make_text(tokenizer, con, res)
                ts.append(t)
                ats.append(a)

            x = torch.cat(ts, dim=0)
            print("x:", x)
            x = x.to(device)

            mask = torch.cat(ats, dim=0)
            print("mask:", mask)
            mask = mask.to(device)

            output = model(x, attention_mask=mask)[0]
            print("model_output:", output)
            y_hat = F.softmax(output, dim=-1)
            print("model_yhat:", y_hat)
            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        print(indice)
        print(probs)
        # |indice| = (len(lines), top_k)

        for i in range(len(responses)):
            sys.stdout.write('%s\t%s\t%s\n' % (
                ' '.join([classes[int(indice[i][j])] for j in range(config.top_k)]), 
                contexts[i],
                responses[i]
            ))

if __name__ == '__main__':
    config = define_argparser()
    main(config)