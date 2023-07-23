from paddleocr import PaddleOCR


def ocr_plate(image, threshold: float=0.95):
    ocr = PaddleOCR(use_gpu=True, show_log=False) # need to run only once to download and load model into memory
    result = ocr.ocr(image)
    for line in result:
        print(line)
    plate_numer = ""
    for _, prediction in result:
        text, score = prediction
        if score < threshold:
            continue
        text = text.replace(".","")
        plate_numer += text
    
    return plate_numer


if __name__ == "__main__":
    image = "cropped.jpg"
    ocr_plate(image)