from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Dataloader:
    def __init__(
        self,
        data_path=Path('/home/miza/Magisterka/src/data/annotations/annotations.csv'),
    ):
        self.data = pd.read_csv(data_path)

    def get_encoder(self):
        self.le = LabelEncoder()
        self.data['Category'] = self.le.fit_transform(self.data['Category'])
        return self.le

    def preprocess_image(self):
        pass

    def get_data_batch(self, batch_size: float = 1):
        y = len(self.data) * batch_size
        if y > len(self.data):
            y = len(self.data)
        return self.data[:y]

    def split_data(self, test_size: float = 0.2):
        train_data, test_data = train_test_split(self.data, test_size=test_size)
        return train_data, test_data
