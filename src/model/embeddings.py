from transformers import BertModel, BertTokenizer
import torch


class BertEmbedding:
    def __init__(self, model_name: str = 'dkleczek/bert-base-polish-uncased-v1'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed(self, batch: list):
        inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=32, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output
