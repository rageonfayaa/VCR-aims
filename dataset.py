from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

@dataclass
class VCRSample:
    index: int
    img_path: Path
    metadata_path: Path
    objects: list[str]
    question: list[Any]
    answer_choices: list[list[Any]]
    rationale_choices: list[list[Any]]
    answer_label: int
    rationale_label: int
    question_text: str | None = None
    answer_choice_texts: list[str] | None = None
    rationale_choice_texts: list[str] | None = None
    image: Image.Image | None = None
    boxes: list[BoundingBox] | None = None

@dataclass
class EvalResult:
    sample_index: int
    predicted_answer: int
    predicted_rationale: int
    gt_answer: int
    gt_rationale: int
    answer_correct: bool
    rationale_correct: bool
    joint_correct: bool
    answer_log_likelihoods: list[float] = field(default_factory=list)
    rationale_log_likelihoods: list[float] = field(default_factory=list)

def detokenize_with_tags(tokens: list[Any], object_names: list[str]) -> str:
    class_counters: dict[str, int] = {}
    index_to_tag: dict[int, str] = {}
    result_parts: list[str] = []
    for token in tokens:
        if isinstance(token, list):
            tag_parts: list[str] = []
            for obj_idx in token:
                if obj_idx in index_to_tag:
                    tag_parts.append(index_to_tag[obj_idx])
                else:
                    obj_class = object_names[obj_idx] if obj_idx < len(object_names) else 'object'
                    class_counters[obj_class] = class_counters.get(obj_class, 0) + 1
                    tag = f'{obj_class}{class_counters[obj_class]}'
                    index_to_tag[obj_idx] = tag
                    tag_parts.append(tag)
            result_parts.append(' and '.join(tag_parts))
        elif isinstance(token, str):
            result_parts.append(token)
        else:
            result_parts.append(str(token))
    return ' '.join(result_parts)

def _convert_hf_tokens(hf_tokens: list[dict]) -> list[Any]:
    result: list[Any] = []
    for token in hf_tokens:
        if token['kind'] == 'text':
            result.append(token['text'])
        elif token['kind'] == 'objects':
            result.append(token['object_indices'])
    return result

def load_metadata(metadata_path: Path) -> list[BoundingBox]:
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    boxes: list[BoundingBox] = []
    for raw_box in meta['boxes']:
        boxes.append(BoundingBox(x1=raw_box[0], y1=raw_box[1], x2=raw_box[2], y2=raw_box[3]))
    return boxes

def load_vcr_dataset(jsonl_path: Path, images_root: Path, num_samples: int | None=None) -> list[VCRSample]:
    samples: list[VCRSample] = []
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if num_samples is not None and idx >= num_samples:
                break
            entry = json.loads(line.strip())
            img_path = images_root / entry['img_fn']
            metadata_path = images_root / entry['metadata_fn']
            if not img_path.exists():
                logger.warning(f'Image not found, skipping sample {idx}: {img_path}')
                continue
            if not metadata_path.exists():
                logger.warning(f'Metadata not found, skipping sample {idx}: {metadata_path}')
                continue
            samples.append(VCRSample(index=idx, img_path=img_path, metadata_path=metadata_path, objects=entry['objects'], question=entry['question'], answer_choices=entry['answer_choices'], rationale_choices=entry['rationale_choices'], answer_label=entry['answer_label'], rationale_label=entry['rationale_label']))
    logger.info(f'Loaded {len(samples)} samples from {jsonl_path.name}')
    return samples

def load_vcr_from_hf(split: str='train', images_root: Path | None=None, num_samples: int | None=None) -> list[VCRSample]:
    from datasets import load_dataset
    logger.info(f'Loading VCR from HuggingFace (Rowan/vcr, split={split})...')
    ds = load_dataset('Rowan/vcr', name='questions', split=split)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    samples: list[VCRSample] = []
    skipped = 0
    for idx, entry in enumerate(ds):
        img_path = images_root / entry['img_fn'] if images_root else Path(entry['img_fn'])
        metadata_path = images_root / entry['metadata_fn'] if images_root else Path(entry['metadata_fn'])
        if images_root and (not img_path.exists()):
            logger.warning(f'Image not found, skipping sample {idx}: {img_path}')
            skipped += 1
            continue
        if images_root and (not metadata_path.exists()):
            logger.warning(f'Metadata not found, skipping sample {idx}: {metadata_path}')
            skipped += 1
            continue
        question = _convert_hf_tokens(entry['question_tokens'])
        answer_choices = [_convert_hf_tokens(choice) for choice in entry['answer_choice_tokens']]
        rationale_choices = [_convert_hf_tokens(choice) for choice in entry['rationale_choice_tokens']]
        samples.append(VCRSample(index=idx, img_path=img_path, metadata_path=metadata_path, objects=entry['objects'], question=question, answer_choices=answer_choices, rationale_choices=rationale_choices, answer_label=entry['answer_label'], rationale_label=entry['rationale_label'], question_text=entry.get('question_text'), answer_choice_texts=entry.get('answer_choice_texts'), rationale_choice_texts=entry.get('rationale_choice_texts')))
    if skipped:
        logger.warning(f'Skipped {skipped} samples due to missing files')
    logger.info(f'Loaded {len(samples)} samples from HuggingFace (Rowan/vcr, {split})')
    return samples
