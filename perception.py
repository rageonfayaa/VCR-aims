from __future__ import annotations
import logging
import torch
from PIL import Image
from ultralytics import YOLO
from src.dataset import BoundingBox
logger = logging.getLogger(__name__)

class PerceptionLayer:

    def __init__(self, yolo_model_name: str='yolo11n.pt', device: torch.device | None=None) -> None:
        self.device = device or torch.device('cpu')
        logger.info(f'Loading YOLO model: {yolo_model_name}')
        self.model = YOLO(yolo_model_name)
        logger.info('✓ YOLO perception layer initialized')

    def crop_from_boxes(self, image: Image.Image, boxes: list[BoundingBox], object_names: list[str]) -> dict[int, Image.Image]:
        crops: dict[int, Image.Image] = {}
        img_width, img_height = image.size
        for idx, (box, obj_name) in enumerate(zip(boxes, object_names)):
            x1 = max(0, min(box.x1, img_width))
            y1 = max(0, min(box.y1, img_height))
            x2 = max(0, min(box.x2, img_width))
            y2 = max(0, min(box.y2, img_height))
            if x2 <= x1 or y2 <= y1:
                logger.debug(f'Skipping degenerate box for object {idx} ({obj_name})')
                continue
            crop = image.crop((x1, y1, x2, y2))
            crops[idx] = crop
        return crops

    def detect_objects(self, image: Image.Image) -> list[BoundingBox]:
        results = self.model(image, verbose=False)
        yolo_boxes: list[BoundingBox] = []
        for box in results[0].boxes.xyxy:
            coords = box.cpu().numpy()
            yolo_boxes.append(BoundingBox(x1=float(coords[0]), y1=float(coords[1]), x2=float(coords[2]), y2=float(coords[3])))
        return yolo_boxes
