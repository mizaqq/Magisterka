from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


class Embedding:
    def embed_with_bert(batch: list, model_name: str = 'dkleczek/bert-base-polish-uncased-v1'):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=32, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output

    def embed_with_mini(sentences: list):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        return embeddings

    def finetune_mini(train_data):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        train_examples = []
        for index, row in train_data.iterrows():
            train_examples.append(InputExample(texts=[row['text'], row['text']], label=int(row['category'])))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        loss = losses.SoftmaxLoss(
            model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=8
        )
        model.fit(train_objectives=[(train_dataloader, loss)], epochs=50, warmup_steps=100)
        model.save('/home/miza/Magisterka/src/model/mini_model')
        return model

    def load_mini_model(path='/home/miza/Magisterka/src/model/mini_model'):
        model = SentenceTransformer(path)
        return model

    def embed_with_given_model(model, sentences: list):
        embeddings = model.encode(sentences)
        return embeddings
