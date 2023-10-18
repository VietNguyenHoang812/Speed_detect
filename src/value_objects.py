from src.abstract.dto import DTO


class PipelineOutputDTO(DTO):
    def __init__(self, image_url: str, BKS: str="Khong phat hien duoc BKS", speed: int=-1) -> None:
        self.image_url = image_url
        self.BKS = BKS
        self.speed = speed
    
    def toResponse(self):
        response = {
            
        }