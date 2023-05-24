from paddleocr import PaddleOCR,draw_ocr
from datetime import datetime
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
start = datetime.now()
ocr = PaddleOCR(use_gpu=False) # need to run only once to download and load model into memory
# img_path = 'img_12.jpg'
img_path = 't1.JPG'
result = ocr.ocr(img_path)
for line in result:
    print(line)


# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path="doc/fonts/latin.ttf")
im_show = Image.fromarray(im_show)
im_show.save('output_imgs/result.jpg')
end = datetime.now()
duration = (end-start).total_seconds()
print(duration)





#python paddleocr.py --use_gpu False --lang en --image_dir doc/imgs_en/img_12.jpg --type ocr --output output_imgs
