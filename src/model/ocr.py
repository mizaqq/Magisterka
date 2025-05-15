from paddleocr import PaddleOCR

from utils.utils import preprocess_cost, preprocess_text


class OCR:
    def __init__(self):
        self.paddleocr = PaddleOCR(
            use_angle_cls=True,
            lang='pl',
            det_algorithm='DB',
            det_box_type='poly',
            rec_algorithm='CRNN',
            rec_char_type='pl',
            rec_batch_num=15,
            use_space_char=True,
            det_db_thresh=0.35,
            det_db_box_thresh=0.75,
            layout=True,
            table=True,
            table_algorithm='TableAttn',
            use_dilation=True,
        )

    def perform_ocr(self, img_path):
        return self.paddleocr.ocr(img_path, cls=False, rec=True, det=True)

    def match_boxes_within_distance(self, boxes, threshold=10):
        matched_groups = []
        while len(boxes) > 0:
            line = []
            label1 = boxes[0][1]
            box1 = boxes[0][0]
            ymin1 = min([point[1] for point in box1])
            ymax1 = max([point[1] for point in box1])
            line = [label1]
            while len(boxes) > 1:
                box2 = boxes[1][0]
                label2 = boxes[1][1]
                ymin2 = min([point[1] for point in box2])
                ymax2 = max([point[1] for point in box2])
                if abs(ymin1 - ymin2) < threshold or abs(ymax1 - ymax2) < threshold:
                    line.append(label2)
                    boxes.pop(0)
                    if len(boxes) > 0:
                        label1 = boxes[0][1]
                        box1 = boxes[0][0]
                        ymin1 = min([point[1] for point in box1])
                        ymax1 = max([point[1] for point in box1])
                else:
                    break
            if len(line) > 1:
                matched_groups.append(' '.join(line))
            boxes.pop(0)
        return matched_groups

    def get_data_from_image(self, img_path):
        ocr_data = self.perform_ocr(img_path)
        boxes = []
        for line in ocr_data:
            for word_info in line:
                box = word_info[0]
                label = word_info[1][0]
                boxes.append((box, label))
        return self.match_boxes_within_distance(boxes)

    def get_preprocessed_data(self, img_path):
        ocr_data = self.get_data_from_image(img_path)
        preprocessed_data = []
        for data in ocr_data:
            text = preprocess_text(data)
            cost = preprocess_cost(data)
            preprocessed_data.append((text, cost))
        return preprocessed_data
