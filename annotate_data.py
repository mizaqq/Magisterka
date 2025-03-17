from pytesseract import pytesseract, Output
import cv2
import pandas as pd
import json
from pathlib import Path
import numpy as np


from utils.utils import (
    grayscale,
    otsu_threshold,
    adaptive_threshold,
    sauvola_threshold,
    gaussian_blur,
    gaussian_blur1,
    gaussian_blur2,
    median_blur,
    sharpen,
    resize,
    contrast_stretching,
    histogram_equalization,
    clahe,
    morphological_closing,
    morphological_opening,
    dilation,
    erosion,
    bilateral_filter,
    unsharp_mask,
)


def load_image(path: Path):
    image = cv2.imread(path)
    img = grayscale(image)
    img = sauvola_threshold(img)
    img = clahe(img)
    img = bilateral_filter(img)
    img = sharpen(img)
    img = gaussian_blur1(img)
    return img


def get_text(image):
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 6 --oem 3', lang='pol_finetuned')
    lines = []
    current_line = []
    prev_line_number = ocr_data['line_num'][0]
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i]
        line_number = ocr_data['line_num'][i]

        if word.strip():
            if line_number != prev_line_number:
                lines.append(" ".join(current_line))
                current_line = []
                prev_line_number = line_number

            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line) if current_line != [''] else "")
    for i in lines:
        if i == '':
            lines.remove(i)
    return lines


def annotate_image(path: Path, lines, annotations_data: list[dict]):
    annotations = []
    indexes = [annotation['index'] for annotation in annotations_data]
    for i, line in enumerate(lines):
        if i in indexes:
            continue
        print(line)
        label = input("Enter label: ")
        print('cost proposition', line[-5:-1])
        cost = input("Enter cost: ")
        if cost == '':
            cost = line[-5:-1]
        annotation = {'img': f'{path.name}.jpg', 'index': i, "label": label, "cost": cost}
        annotations.append(annotation)
    with open(f'/home/miza/Magisterka/src/data/annotations/{path.name}.jsonl', 'w', encoding='utf-8') as f:
        for entry in annotations:
            if entry['img'] not in annotations_data:
                f.write(json.dumps(entry))
                f.write('\n')


"""
Food
Housing
Transportation
Telecom
Healthcare
Clothing
Entertainment
Finance
Miscellaneous
Other
Discount
"""
if __name__ == '__main__':
    images = list(Path('/home/miza/Magisterka/src/data/images').glob('*.jpg'))
    for image in images:
        img = load_image(image)
        lines = get_text(img)
        annotation_path = Path('/home/miza/Magisterka/src/data/annotations') / f'{image.stem}.jsonl'
        if annotation_path.exists():
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations = json.loads(f.read())
        else:
            annotations = []
        annotate_image(annotation_path, lines, annotations)
