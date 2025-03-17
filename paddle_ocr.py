from paddleocr import PaddleOCR
import paddle

# Initialize PaddleOCR (English by default)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=2.5,
    rec_algorithm='CRNN',
    rec_batch_num=6,
    use_space_char=True,
)  # need to specify use_angle_cls=True if you want to detect rotated text

# Path to your image
img_path = '/home/miza/Magisterka/src/data/images/1.jpg'

# Run OCR
result = ocr.ocr(
    img_path,
    cls=True,
)

# Print OCR results
for line in result[0]:
    print(f"Detected text: {line[1][0]} | Confidence: {line[0]}")
