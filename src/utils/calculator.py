import math


def calculate_distance(ori_center, crop_center):
    ho, wo = ori_center
    hcrop, wcrop = crop_center
    return math.sqrt((ho-hcrop)**2+(wo-wcrop)**2)