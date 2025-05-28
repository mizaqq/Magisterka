from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import EarlyStoppingCallback

from torch.utils.data import DataLoader
import random
from itertools import combinations
from typing import List, Optional

import pandas as pd


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

    def finetune_mini(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        path='/home/miza/Magisterka/src/model/mini_model',
        max_pos_per_cat: int = 200,
        neg_pos_ratio: float = 0.7,
        batch_size: int = 16,
        epochs: int = 50,
        warmup_steps: int = 100,
    ) -> SentenceTransformer:
        train_examples = Embedding._sample_pairs(train_data, max_pos_per_cat, neg_pos_ratio)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        evaluator = None
        dev_examples = Embedding._sample_pairs(val_data, max_pos_per_cat, neg_pos_ratio)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name="dev")
        loss = losses.ContrastiveLoss(
            model=model,
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            margin=0.5,
        )
        model.fit(
            train_objectives=[(train_dataloader, loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluation_steps=len(train_dataloader),
            output_path=path,
            save_best_model=True,
        )
        return model

    def load_mini_model(path='/home/miza/Magisterka/src/model/mini_model'):
        model = SentenceTransformer(path)
        return model

    def embed_with_given_model(model, sentences: list):
        embeddings = model.encode(sentences)
        return embeddings

    def _sample_pairs(
        df: pd.DataFrame,
        max_pos_per_cat: int = 200,
        neg_pos_ratio: float = 1.0,
    ) -> List[InputExample]:
        groups = df.groupby("category")["text"].apply(list).to_dict()
        all_texts = df["text"].tolist()
        text2cat = df.set_index("text")["category"].to_dict()

        pos_pairs = []
        for cat, texts in groups.items():
            pairs = list(combinations(texts, 2))
            sampled = random.sample(pairs, min(len(pairs), max_pos_per_cat))
            pos_pairs.extend(sampled)

        neg_pairs = []
        n_neg = int(len(pos_pairs) * neg_pos_ratio)
        while len(neg_pairs) < n_neg:
            t1 = random.choice(all_texts)
            t2 = random.choice(all_texts)
            if text2cat[t1] != text2cat[t2]:
                neg_pairs.append((t1, t2))

        examples = []
        for t1, t2 in pos_pairs:
            examples.append(InputExample(texts=[t1, t2], label=1.0))
        for t1, t2 in neg_pairs:
            examples.append(InputExample(texts=[t1, t2], label=0.0))

        random.shuffle(examples)
        return examples


class BertModelWrapper:
    def __init__(self, model_name: str = 'dkleczek/bert-base-polish-uncased-v1'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, sentences: list):
        inputs = self.tokenizer(sentences, return_tensors='pt', truncation=True, max_length=32, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output
