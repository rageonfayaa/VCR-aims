from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from src.dataset import BoundingBox
logger = logging.getLogger(__name__)

@dataclass
class VCRItem:
    image: Image.Image
    question_text: str
    answer_choices: list[str]
    rationale_choices: list[str]
    answer_label: int
    rationale_label: int
    objects: list[str]
    boxes: list[BoundingBox]
    img_fn: str
    sample_index: int

class VCRStreamDataset(IterableDataset):

    def __init__(self, split: str='train', max_samples: int | None=None) -> None:
        super().__init__()
        self.split = split
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[VCRItem]:
        from datasets import load_dataset
        logger.info(f'Streaming VCR from HuggingFace (Rowan/vcr image_examples, split={self.split})...')
        hf_split = 'validation' if self.split == 'val' else self.split
        ds = load_dataset('Rowan/vcr', name='image_examples', split=hf_split, streaming=True)
        sample_count = 0
        for row in ds:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break
            image: Image.Image = row['image']
            img_fn: str = row['img_fn']
            objects: list[str] = row['objects']
            boxes: list[BoundingBox] = []
            for raw_box in row['boxes']:
                boxes.append(BoundingBox(x1=float(raw_box[0]), y1=float(raw_box[1]), x2=float(raw_box[2]), y2=float(raw_box[3])))
            for annot in row['annotations']:
                if self.max_samples is not None and sample_count >= self.max_samples:
                    break
                yield VCRItem(image=image, question_text=annot.get('question_text', ''), answer_choices=annot.get('answer_choice_texts', []), rationale_choices=annot.get('rationale_choice_texts', []), answer_label=annot['answer_label'], rationale_label=annot['rationale_label'], objects=objects, boxes=boxes, img_fn=img_fn, sample_index=sample_count)
                sample_count += 1
        logger.info(f'Streamed {sample_count} VCR samples from HuggingFace')

def collate_vcr(batch: list[VCRItem]) -> dict[str, Any]:
    return {'images': [item.image for item in batch], 'questions': [item.question_text for item in batch], 'answer_choices': [item.answer_choices for item in batch], 'rationale_choices': [item.rationale_choices for item in batch], 'answer_labels': torch.tensor([item.answer_label for item in batch]), 'rationale_labels': torch.tensor([item.rationale_label for item in batch]), 'objects': [item.objects for item in batch], 'boxes': [item.boxes for item in batch], 'img_fns': [item.img_fn for item in batch], 'sample_indices': torch.tensor([item.sample_index for item in batch])}

def create_vcr_dataloader(split: str='train', batch_size: int=1, max_samples: int | None=None, num_workers: int=0) -> DataLoader:
    dataset = VCRStreamDataset(split=split, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_vcr, num_workers=num_workers)
