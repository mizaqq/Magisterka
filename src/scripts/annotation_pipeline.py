import csv
import os

from src.model.classifier import XGBoost
from src.model.dataloader import Dataloader
from src.model.embeddings import Embedding
from src.model.ocr import OCR

ANNOTATIONS_PATH = '/home/miza/Magisterka/src/data/annotations/annotations_gpt.csv'
IMAGES = '/home/miza/Magisterka/src/data/images/gpt/'


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
    if not os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['img', 'product', 'label', 'cost'],
            )
            writer.writeheader()
        annotations_data = []
    else:
        with open(ANNOTATIONS_PATH, 'r', newline='') as f:
            reader = csv.DictReader(f)
            annotations_data = [row for row in reader]
    images = os.listdir(IMAGES)
    for i in annotations_data:
        if i['img'] in images:
            images.remove(i['img'])
    return images


def get_baseline_model():
    dataloader = Dataloader()
    dataloader.get_encoder()
    data = dataloader.get_data_batch()
    embeddings = Embedding.embed_with_bert(data['OCR_product'].tolist())
    model = XGBoost(learning_rate=0.1, n_estimators=100, max_depth=3)
    model.fit(embeddings, data['category'].tolist())
    return model, dataloader.le


def annotation_pipeline(products, img, classes):
    annotations = []
    for k, v in products.items():
        category, cost_returned = annotation_menu(k, v[0], v[1], classes)
        row = {
            'img': img,
            'product': k,
            'label': category,
            'cost': cost_returned,
        }
        print(f"Product: {k}, Category: {category}, Cost: {cost_returned}")
        annotations.append(row)
    return annotations


def save_annotations(annotations):
    if not os.path.exists(ANNOTATIONS_PATH):
        mode = 'w'
    else:
        mode = 'a'
    with open(ANNOTATIONS_PATH, mode, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['img', 'product', 'label', 'cost'],
        )
        if mode == 'w':
            writer.writeheader()
        else:
            pass
        for row in annotations:
            writer.writerow(row)


def main():
    model, label_encoder = get_baseline_model()
    images = get_unannotated_images()
    ocr = OCR()
    for img in images:
        result = ocr.get_data_from_image(IMAGES + img)
        result_clean = ocr.get_preprocessed_data(IMAGES + img)
        pred_data = Embedding.embed_with_bert([r[0] for r in result_clean])
        preds = model.predict(pred_data)
        labels = label_encoder.inverse_transform(preds)
        products = {
            r[0]: [r[1], r[2][1]] for r in zip(result, labels, result_clean)
        }  # r[0] == product name,r[1]==category, r[2][1] == cost

        classes = [c for c in label_encoder.classes_]
        print('File', img, ' Categories', [f"{i + 1}:{c}" for i, c in enumerate(classes)])

        annotations = annotation_pipeline(products, img, classes)
        save_annotations(annotations)
    print("Annotations saved successfully.")


main()
