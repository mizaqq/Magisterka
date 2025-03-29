import re

# for text pre-processing
import string

import pandas as pd
import torch
from paddleocr import PaddleOCR
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
from xgboost import XGBClassifier

tokenizer = BertTokenizer.from_pretrained('dkleczek/bert-base-polish-uncased-v1')
bert_model = BertModel.from_pretrained('dkleczek/bert-base-polish-uncased-v1')


def data_preprocess(df):
    df['preprocessed_text'] = df['OCR_product'].apply(lambda x: preprocess_text(x))
    df['cost'] = df['OCR_product'].apply(lambda x: preprocess_cost(x))
    embedings = calculate_embedings(df['preprocessed_text'].tolist())
    le = LabelEncoder()
    categories = le.fit_transform(df['Category'])
    return df, embedings, categories


def calculate_embedings(df):
    inputs = tokenizer(df, return_tensors='pt', truncation=True, max_length=32, padding='max_length')
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return outputs.pooler_output


def preprocess_text(text):
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text.strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\b[a-zA-Z]\b\s*', ' ', text).strip()
    return text


def preprocess_cost(text):
    text = text.lower()
    text = text.strip()
    # text = re.sub(r'\s+', '', text)
    pattern = re.compile(r'(\d{1,3}\.\d{2})\d?')
    return pattern.findall(text)


def clean_token(token):
    for t in token:
        if len(t) == 1:
            print(t)
            token.remove(t)
    return token


def get_baseline_model(df):
    embedings, categories = data_preprocess(df)
    model = XGBClassifier()
    model.fit(embedings, categories)
    return model


def ocr_data(img_path):
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='pl',
        rec_algorithm='CRNN',
        rec_batch_num=6,
        use_space_char=True,
    )

    return ocr.ocr(
        img_path,
        cls=True,
    )


def parse_ocr():
    pass


def main():
    ocr = ocr_data('/home/miza/magisterka/Magisterka/src/data/images/5.jpg')
    baseline_data = pd.read_csv("/home/miza/magisterka/Magisterka/src/data/bert.csv")
    baseline_mode = get_baseline_model(baseline_data)


main()
