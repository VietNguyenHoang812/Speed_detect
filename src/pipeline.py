import time

from src.stages.plate_detection import LicensePlateDetection
from src.stages.plate_ocr import LicensePlateOCR
from src.stages.speed_detect import SpeedDetection
from value_objects import PipelineOutputDTO


def pipeline(image_url: str):
    image_url = image_url.replace("%2F", "/")
    start_time = time.time()
    first_stage_result = LicensePlateDetection().run(image_url)
    second_stage_result = LicensePlateOCR().run(first_stage_result)
    third_stage_result = SpeedDetection().run(image_url)
    prediction_time = time.time() - start_time

    output = PipelineOutputDTO(image_url, second_stage_result, third_stage_result)
    response = output.toResponse(prediction_time)
    return response
