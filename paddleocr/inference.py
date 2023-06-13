from paddleocr import PaddleOCR


ocr = PaddleOCR(use_gpu=True) # need to run only once to download and load model into memory
img_path = 'crop/test.jpg'
result = ocr.ocr(img_path)
for line in result:
    print(line)