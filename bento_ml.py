import bentoml
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.adapters import JsonInput
import time
import torch
import torch.nn.functional as F

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

@bentoml.env(pip_packages=["transformers==4.8.2", "torch==1.9.0"])
@bentoml.artifacts([TransformersModelArtifact('distilbert')])
class TransformerService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), mb_max_latency=50)
    def predict(self, parsed_json):
        with torch.no_grad():
            model = self.artifacts.distilbert.get("model")
            tokenizer = self.artifacts.distilbert.get("tokenizer")
            model.cuda(0)
            device = next(model.parameters()).device

            context = parsed_json['context']
            response = parsed_json['response']
            start = time.time()
            encoding = tokenizer(
                text=context,
                text_pair=response,
                max_length=512,
                truncation=True,
                padding=True,
                pad_to_max_length=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
            mask = encoding['attention_mask']
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            output = model(input_ids, mask)[0]
            y_hat = F.softmax(output, dim=-1).cpu()
        return y_hat, time.time() - start


saved_data = torch.load(
    './models/tri_aug.pth',
    map_location = 'cuda:0'
)
train_config = saved_data['config']
bert_best = saved_data['bert']
classes = saved_data['classes']

ts = TransformerService()

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(classes))
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', strip_accents=False)
model.load_state_dict(bert_best)

artifact = {"model":model, "tokenizer":tokenizer}
ts.pack("distilbert", artifact)
saved_path = ts.save()
