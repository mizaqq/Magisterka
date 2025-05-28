from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.model.embeddings import Embedding
from copy import deepcopy
from sklearn.utils import resample
from src.utils.utils import preprocess_cost, preprocess_text
import numpy as np


class Dataloader:
    def __init__(self, data_path=Path('/home/miza/Magisterka/src/data/annotations/annotations_6classes.csv')):
        self.data = pd.read_csv(data_path).drop(columns=['img'])
        self.get_encoder()

    def get_encoder(self):
        self.le = LabelEncoder()
        self.le.fit(self.data['category'])
        return self.le

    def preprocess_image(self):
        pass

    def get_preprocessed_data(self):
        data = pd.DataFrame()
        data['text'] = self.data['OCR_product'].apply(preprocess_text).tolist()
        data['cost'] = self.data['OCR_product'].apply(preprocess_cost).tolist()
        data['category'] = self.le.transform(self.data['category'])
        return data

    def get_data_batch(self, batch_size: float = 1):
        data = deepcopy(self.data)
        data['category'] = self.le.transform(data['category'])
        y = len(data) * batch_size
        if y > len(data):
            y = len(data)
        return data[:y]

    def split_data(self, data=None, test_size: float = 0.25):
        if data is None:
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, stratify=self.data['category'], random_state=42
            )
        else:
            train_data, test_data = train_test_split(
                data, test_size=test_size, stratify=self.data['category'], random_state=42
            )

        train_data, val_data = train_test_split(
            train_data, test_size=0.1, stratify=train_data['category'], random_state=42
        )
        return train_data, test_data, val_data

    def get_training_ready_data(
        self,
        model,
        split_size: float = 0.25,
    ):
        train_data, test_data, val_data = self.split_data(None, split_size)
        train_batch = self.data_to_embeddings(train_data, model)
        test_batch = self.data_to_embeddings(test_data, model)
        val_batch = self.data_to_embeddings(val_data, model)
        return train_batch, test_batch, val_batch

    def data_to_embeddings(self, data, model):
        data['Text'] = data['OCR_product'].apply(preprocess_text)
        embeddings = Embedding.embed_with_given_model(model, data['Text'].tolist())
        categories = self.le.transform(data['category'])
        return {
            'embeddings': embeddings,
            'category': categories,
        }

    def get_gpt_data_for_training(
        self, path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv')
    ):
        gpt_data = pd.read_csv(path)
        gpt_data['text'] = gpt_data['OCR_product'].apply(preprocess_text).tolist()
        gpt_data['cost'] = gpt_data['OCR_product'].apply(preprocess_cost).tolist()
        gpt_data['category'] = self.le.transform(gpt_data['category'])
        return gpt_data

    def get_gpt_data_training_ready_data(
        self,
        model,
        path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv'),
        balance: bool = True,
    ):
        gpt_data = pd.read_csv(path)
        if balance:
            gpt_data = self.balance_gpt_data(gpt_data)
        gpt_categories = self.le.transform(gpt_data['category'])
        gpt_data['Text'] = gpt_data['OCR_product'].apply(preprocess_text)
        gpt_embeddings = Embedding.embed_with_given_model(model, gpt_data['Text'].tolist())
        gpt_batch = {
            'embeddings': gpt_embeddings,
            'category': gpt_categories,
        }
        return gpt_batch

    def balance_gpt_data(self, df_gpt):
        return pd.concat(
            [
                resample(
                    df_gpt[df_gpt['category'] == cat],
                    replace=True,
                    n_samples=int(len(self.data) * frac),
                    random_state=42,
                )
                for cat, frac in self.data['category'].value_counts(normalize=True).items()
            ]
        )

    def get_joined_data(self, train_embeddings, train_categories, gpt_embeddings, gpt_categories):
        joined_embeddings = np.concatenate((train_embeddings, gpt_embeddings), axis=0)
        joined_categories = np.append(train_categories, gpt_categories)
        return {
            'embeddings': joined_embeddings,
            'category': joined_categories,
        }
