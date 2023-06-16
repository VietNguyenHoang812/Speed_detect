from paddleocr import PaddleOCR


def predict_plate(image, threshold: float=0.95):
    ocr = PaddleOCR(use_gpu=False, show_log=False) # need to run only once to download and load model into memory
    result = ocr.ocr(image)
    plate_numer = ""
    for _, prediction in result:
        text, score = prediction
        if score < threshold:
            continue
        text = text.replace(".","")
        plate_numer += text
    
    return plate_numer