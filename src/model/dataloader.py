from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.model.embeddings import BertEmbedding
from copy import deepcopy
from sklearn.utils import resample


class Dataloader:
    def __init__(self, data_path=Path('/home/miza/Magisterka/src/data/annotations/annotations_8classes.csv')):
        self.data = pd.read_csv(data_path).drop(columns=['img'])
        self.get_encoder()

    def get_encoder(self):
        self.le = LabelEncoder()
        self.le.fit(self.data['Category'])
        return self.le

    def preprocess_image(self):
        pass

    def get_data_batch(self, batch_size: float = 1):
        data = deepcopy(self.data)
        data['Category'] = self.le.transform(data['Category'])
        y = len(data) * batch_size
        if y > len(data):
            y = len(data)
        return data[:y]

    def split_data(self, test_size: float = 0.2):
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        return train_data, test_data

    def get_training_ready_data(
        self,
        split_size: float = 0.2,
    ):
        train_data, test_data = self.split_data(split_size)
        train_categories = self.le.transform(train_data['Category'])
        test_categories = self.le.transform(test_data['Category'])
        tokenizer = BertEmbedding()
        train_embeddings = tokenizer.embed(train_data['OCR_product'].tolist())
        test_embeddings = tokenizer.embed(test_data['OCR_product'].tolist())
        train_batch = {
            'embeddings': train_embeddings,
            'Category': train_categories,
        }
        test_batch = {
            'embeddings': test_embeddings,
            'Category': test_categories,
        }

        return train_batch, test_batch

    def get_gpt_data_for_finetune(
        self, path: Path = Path('/home/miza/Magisterka/src/data/annotations/gpt_generated_data8classes.csv')
    ):
        gpt_data = pd.read_csv(path)
        gpt_categories = self.le.transform(gpt_data['Category'])
        tokenizer = BertEmbedding()
        gpt_embeddings = tokenizer.embed(gpt_data['OCR_product'].tolist())
        gpt_batch = {
            'embeddings': gpt_embeddings,
            'Category': gpt_categories,
        }
        return gpt_batch

    def balance_gpt_data(self, df_gpt):
        return pd.concat(
            [
                resample(
                    df_gpt[df_gpt['Category'] == cat],
                    replace=True,
                    n_samples=int(len(self.data) * frac),
                    random_state=42,
                )
                for cat, frac in self.data['Category'].value_counts(normalize=True).items()
            ]
        )
