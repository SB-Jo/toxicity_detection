import torch
from torch.utils.data import Dataset

class TextClassificationCollator():
    def __init__(self, tokenizer, max_length, with_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        contexts = [s['context'] for s in samples]
        responses = [s['response'] for s in samples]
        targets = [s['target'] for s in samples]

        # encoding = self.tokenizer(
        #     text = contexts,
        #     text_pair = responses,
        #     padding=True,
        #     truncation=True,
        #     return_tensors='pt',
        #     max_length=self.max_length
        # )

        encoding = {}

        def make_text(context, response):
            con = self.tokenizer(context, max_length=128, truncation=True, return_tensors='pt')
            res = self.tokenizer(response, max_length=512, truncation=True, return_tensors='pt')
            text = torch.cat([con['input_ids'], res['input_ids']], dim=-1)
            attention_mask = torch.cat([con['attention_mask'], res['attention_mask']], dim=-1)
            pad_length = self.max_length - text.size(1)
            if pad_length > 0:
                zero_pad = torch.tensor([0]*pad_length).unsqueeze(0)
                text = torch.cat([text, zero_pad], dim=-1)
                attention_mask = torch.cat([attention_mask, zero_pad], dim=-1)
            
            text = text[:512]
            attention_mask = attention_mask[:512]
            return text, attention_mask

        ts, ats = [], []
        for con, res in zip(contexts, responses):
            t, a = make_text(con, res)
            ts.append(t)
            ats.append(a)

        ts = torch.cat([*ts], dim=0)
        ats = torch.cat([*ats], dim=0)

        encoding['input_ids'] = ts
        encoding['attention_mask'] = ats



        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(targets, dtype=torch.long)
        }

        if self.with_text:
            return_value['text'] = [con + '[sep]'+res for con, res in zip(contexts, responses)]

        return return_value

class TextClassificationDataset(Dataset):
    def __init__(self, contexts, responses, targets):
        self.contexts = contexts
        self.responses = responses
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = str(self.contexts[item])
        response = str(self.responses[item])
        target = self.targets[item]

        return {
            'context':context,
            'response':response,
            'target':target,
        }