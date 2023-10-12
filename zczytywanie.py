from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

os.chdir('C:\\Users\\zarbe\\OneDrive\\Pulpit\\Projekt python')
x=Image.open('paragon.jpg')
paragon= pytesseract.image_to_string(x, lang='pol')
print(paragon)

