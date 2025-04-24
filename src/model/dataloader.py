from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.model.embeddings import Embedding
from copy import deepcopy
from sklearn.utils import resample
from src.utils.utils import preprocess_cost, preprocess_text


class Dataloader:
    def __init__(self, data_path=Path('/home/miza/Magisterka/src/data/annotations/annotations_8classes.csv')):
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

    def split_data(self, data=None, test_size: float = 0.2):
        if data is None:
            train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        else:
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        return train_data, test_data

    def get_training_ready_data(
        self,
        split_size: float = 0.2,
    ):
        train_data, test_data = self.split_data(None, split_size)
        train_categories = self.le.transform(train_data['category'])
        test_categories = self.le.transform(test_data['category'])
        train_data['Text'] = train_data['OCR_product'].apply(preprocess_text)
        test_data['Text'] = test_data['OCR_product'].apply(preprocess_text)
        train_embeddings = Embedding.embed_with_bert(train_data['Text'].tolist())
        test_embeddings = Embedding.embed_with_bert(test_data['Text'].tolist())
        train_batch = {
            'embeddings': train_embeddings,
            'category': train_categories,
        }
        test_batch = {
            'embeddings': test_embeddings,
            'category': test_categories,
        }
        return train_batch, test_batch

    def get_gpt_data_for_finetune(
        self, path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv')
    ):
        gpt_data = pd.read_csv(path)
        gpt_categories = self.le.transform(gpt_data['category'])
        gpt_embeddings = Embedding.embed_with_bert(gpt_data['OCR_product'].tolist())
        gpt_batch = {
            'embeddings': gpt_embeddings,
            'category': gpt_categories,
        }
        return gpt_batch

    def get_gpt_data_for_finetune_with_balance(
        self, path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv')
    ):
        gpt_data = pd.read_csv(path)
        balanced_gpt_data = self.balance_gpt_data(gpt_data)
        gpt_categories = self.le.transform(balanced_gpt_data['category'])
        balanced_gpt_data['Text'] = balanced_gpt_data['OCR_product'].apply(preprocess_text)
        gpt_embeddings = Embedding.embed_with_bert(balanced_gpt_data['Text'].tolist())
        gpt_batch = {
            'embeddings': gpt_embeddings,
            'category': gpt_categories,
        }
        return gpt_batch

    def get_gpt_data_for_finetune_with_balance_and_mini(
        self, model, path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv')
    ):
        gpt_data = pd.read_csv(path)
        balanced_gpt_data = self.balance_gpt_data(gpt_data)
        gpt_categories = self.le.transform(balanced_gpt_data['category'])

        balanced_gpt_data['category'] = self.le.transform(balanced_gpt_data['category'])
        balanced_gpt_data['Text'] = balanced_gpt_data['OCR_product'].apply(preprocess_text)
        gpt_embeddings = Embedding.embed_with_given_model(model, balanced_gpt_data['Text'].tolist())
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
