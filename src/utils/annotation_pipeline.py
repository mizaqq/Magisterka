import re

# for text pre-processing
import string

import pandas as pd
import torch
from paddleocr import PaddleOCR
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
from xgboost import XGBClassifier
import csv
import os

tokenizer = BertTokenizer.from_pretrained('dkleczek/bert-base-polish-uncased-v1')
bert_model = BertModel.from_pretrained('dkleczek/bert-base-polish-uncased-v1')


def data_preprocess(df):
    df['preprocessed_text'] = df['OCR_product'].apply(lambda x: preprocess_text(x))
    df['cost'] = df['OCR_product'].apply(lambda x: preprocess_cost(x))
    embedings = calculate_embedings(df['preprocessed_text'].tolist())
    return df, embedings


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


def get_baseline_model(df, categories):
    _, embedings = data_preprocess(df)
    model = XGBClassifier()
    model.fit(embedings, categories)
    return model


def ocr_data(paddleocr: PaddleOCR, img_path):
    return paddleocr.ocr(img_path, cls=False, rec=True, det=True)


def match_boxes_within_distance(boxes, threshold=19):
    matched_groups = []

    grouped = [False] * len(boxes)

    for i in range(len(boxes)):
        if grouped[i]:
            continue

        group = [boxes[i]]
        label1 = boxes[i][1]  # Label of box 1
        box1 = boxes[i][0]  # Coordinates of box 1

        ymin1 = min([point[1] for point in box1])  # Find ymin for box 1
        ymax1 = max([point[1] for point in box1])  # Find ymax for box 1

        for j in range(i + 1, len(boxes)):
            if grouped[j]:  # Skip already grouped boxes
                continue
            box2 = boxes[j][0]  # Coordinates of box 2
            label2 = boxes[j][1]  # Label of box 2
            ymin2 = min([point[1] for point in box2])  # Find ymin for box 2
            ymax2 = max([point[1] for point in box2])  # Find ymax for box 2

            if abs(ymin1 - ymin2) < threshold or abs(ymax1 - ymax2) < threshold:
                group.append(boxes[j])
                grouped[j] = True

        group.sort(key=lambda box: min([point[0] for point in box[0]]))  # Sort by xmin
        line = [box[1][0] for box in group]
        matched_groups.append(line)

    return matched_groups


def make_prediction(result, model):
    preprocessed_text = [preprocess_text(i[0]) for i in result]
    preprocessed_cost = [preprocess_cost(i[1]) for i in result]
    embeddings = calculate_embedings(preprocessed_text)
    predictions = model.predict(embeddings)
    return predictions, preprocessed_cost


def annotation_menu(product, pred, cost, classes):
    print('Product name: ', product)
    print("Proposed category: ", pred)
    choice = input("Enter your choice(y to accept): ")
    if choice == 'y':
        category = pred
        print("Accepted")
    else:
        category = classes[int(choice) - 1]
    print("Proposed cost", [f"{i + 1}:{c}" for i, c in enumerate(cost)])
    choice = input("Enter your cost: ")
    if choice == '1':
        cost = cost[int(choice) - 1]
    elif choice == '2':
        cost = cost[int(choice) - 1]
    elif choice == '3':
        cost = cost[int(choice) - 1]
    else:
        cost = float(choice)
    return category, cost


def get_unannotated_images():
    if not os.path.exists('/home/miza/Magisterka/src/data/annotations/annotations.csv'):
        with open('/home/miza/Magisterka/src/data/annotations/annotations.csv', 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['img', 'product', 'label', 'cost'],
            )
            writer.writeheader()
        annotations_data = []
    else:
        with open('/home/miza/Magisterka/src/data/annotations/annotations.csv', 'r', newline='') as f:
            reader = csv.DictReader(f)
            annotations_data = [row for row in reader]
    images = os.listdir('/home/miza/Magisterka/src/data/images')
    for i in annotations_data:
        if i['img'] in images:
            images.remove(i['img'])
    return images


def main():
    baseline_data = pd.read_csv("/home/miza/Magisterka/src/data/annotations/annotations.csv")
    le = LabelEncoder()
    categories = le.fit_transform(baseline_data['Category'])
    baseline_model = get_baseline_model(baseline_data, categories)
    images = get_unannotated_images()
    paddleocr = PaddleOCR(
        use_angle_cls=True,
        lang='pl',
        det_algorithm='DB',
        det_box_type='poly',
        rec_algorithm='CRNN',
        rec_char_type='pl',
        rec_batch_num=15,
        use_space_char=True,
        det_db_thresh=0.38,
        det_db_box_thresh=0.75,
        layout=True,
        table=True,
        table_algorithm='TableAttn',
        use_dilation=True,
    )

    for img in images:
        ocr = ocr_data(paddleocr, f'/home/miza/Magisterka/src/data/images/{img}')
        result = match_boxes_within_distance(ocr[0])
        preds, cost = make_prediction(result, baseline_model)
        x = le.inverse_transform(preds)
        classes = [c for c in le.classes_]
        print('Categories', [f"{i + 1}:{c}" for i, c in enumerate(classes)])
        if not os.path.exists('/home/miza/Magisterka/src/data/annotations/annotations.csv'):
            mode = 'w'
        else:
            mode = 'a'
        with open('/home/miza/Magisterka/src/data/annotations/annotations.csv', mode, newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['img', 'product', 'label', 'cost'],
            )
            if mode == 'w':
                writer.writeheader()
            else:
                pass
            for i, product in enumerate(result):
                category, cost_returned = annotation_menu(product, x[i], cost[i], classes)
                row = {
                    'img': img,
                    'product': product,
                    'label': category,
                    'cost': cost_returned,
                }
                print(f"Product: {product[0]}, Category: {category}, Cost: {cost_returned}")
                writer.writerow(row)


main()
