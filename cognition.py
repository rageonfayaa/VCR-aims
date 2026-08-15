from __future__ import annotations
import gc
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
VCR_SYSTEM_PROMPT = 'You are an expert at visual commonsense reasoning. You are shown a scene image from a movie along with cropped close-ups of specific people or objects in that scene. Each crop is labeled with a tag like [person0], [person1], etc. These tags also appear in the question and answer choices so you know exactly which person or object is being referred to.\n\nYour task is to carefully analyze the visual evidence in the images — body language, facial expressions, clothing, spatial relationships, and surrounding context — to select the most plausible answer or rationale from the given choices.'

class CognitionLayer:

    def __init__(self, model_name: str='Qwen/Qwen2.5-VL-7B-Instruct', device: torch.device | None=None) -> None:
        self.device = device or torch.device('cpu')
        logger.info(f'Loading Qwen2.5-VL model: {model_name}')
        logger.info(f'  Target device: {self.device}')
        logger.info(f'  Precision: float16 (half-precision)')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        logger.info('✓ Qwen2.5-VL cognition layer initialized')

    def _build_messages(self, scene_image: Image.Image, crops: dict[int, Image.Image], object_names: list[str], question_text: str, candidate_text: str) -> list[dict]:
        content: list[dict] = []
        content.append({'type': 'image', 'image': scene_image})
        MAX_CROPS = 5
        crop_descriptions: list[str] = []
        for obj_idx, crop_img in sorted(crops.items())[:MAX_CROPS]:
            content.append({'type': 'image', 'image': crop_img})
            obj_class = object_names[obj_idx] if obj_idx < len(object_names) else 'object'
            tag = f'[{obj_class}{obj_idx}]'
            crop_descriptions.append(f'Image {len(crop_descriptions) + 2}: close-up of {tag} (a {obj_class})')
        crop_desc_text = '\n'.join(crop_descriptions)
        prompt_text = f'Image 1: Full scene showing all characters and surroundings.\n{crop_desc_text}\n\nBased on the visual evidence in the images above:\nQuestion: {question_text}\nAnswer: {candidate_text}'
        content.append({'type': 'text', 'text': prompt_text})
        return [{'role': 'system', 'content': [{'type': 'text', 'text': VCR_SYSTEM_PROMPT}]}, {'role': 'user', 'content': content}]

    def _build_prompt_only_messages(self, scene_image: Image.Image, crops: dict[int, Image.Image], object_names: list[str], question_text: str) -> list[dict]:
        content: list[dict] = []
        content.append({'type': 'image', 'image': scene_image})
        MAX_CROPS = 5
        crop_descriptions: list[str] = []
        for obj_idx, crop_img in sorted(crops.items())[:MAX_CROPS]:
            content.append({'type': 'image', 'image': crop_img})
            obj_class = object_names[obj_idx] if obj_idx < len(object_names) else 'object'
            tag = f'[{obj_class}{obj_idx}]'
            crop_descriptions.append(f'Image {len(crop_descriptions) + 2}: close-up of {tag} (a {obj_class})')
        crop_desc_text = '\n'.join(crop_descriptions)
        prompt_text = f'Image 1: Full scene showing all characters and surroundings.\n{crop_desc_text}\n\nBased on the visual evidence in the images above:\nQuestion: {question_text}\nAnswer:'
        content.append({'type': 'text', 'text': prompt_text})
        return [{'role': 'system', 'content': [{'type': 'text', 'text': VCR_SYSTEM_PROMPT}]}, {'role': 'user', 'content': content}]

    def compute_candidate_log_likelihood(self, scene_image: Image.Image, crops: dict[int, Image.Image], object_names: list[str], question_text: str, candidate_text: str, length_normalize: bool=True) -> float:
        full_messages = self._build_messages(scene_image, crops, object_names, question_text, candidate_text)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        image_inputs_full, video_inputs_full = process_vision_info(full_messages)
        full_inputs = self.processor(text=[full_text], images=image_inputs_full, videos=video_inputs_full, return_tensors='pt', padding=True).to(self.device)
        full_ids = full_inputs.input_ids
        L_full = full_ids.shape[1]
        prompt_messages = self._build_prompt_only_messages(scene_image, crops, object_names, question_text)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=False)
        image_inputs_prompt, video_inputs_prompt = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(text=[prompt_text], images=image_inputs_prompt, videos=video_inputs_prompt, return_tensors='pt', padding=True).to(self.device)
        L_prompt = prompt_inputs.input_ids.shape[1]
        num_answer_tokens = L_full - L_prompt
        if num_answer_tokens <= 0:
            logger.warning(f'Candidate produced 0 answer tokens. L_full={L_full}, L_prompt={L_prompt}. Returning -inf.')
            return float('-inf')
        with torch.no_grad():
            outputs = self.model(**full_inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        answer_log_probs = log_probs[:, L_prompt - 1:L_full - 1, :]
        answer_token_ids = full_ids[:, L_prompt:L_full]
        per_token_scores = torch.gather(answer_log_probs, dim=2, index=answer_token_ids.unsqueeze(-1)).squeeze(-1)
        total_log_likelihood = per_token_scores.sum().item()
        if length_normalize:
            total_log_likelihood /= num_answer_tokens
        del full_inputs, prompt_inputs, outputs, logits, log_probs
        del answer_log_probs, answer_token_ids, per_token_scores
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_log_likelihood

    def score_multiple_choice(self, scene_image: Image.Image, crops: dict[int, Image.Image], object_names: list[str], question_text: str, choices: list[str], length_normalize: bool=True) -> tuple[int, list[float]]:
        scores: list[float] = []
        for choice_idx, choice_text in enumerate(choices):
            score = self.compute_candidate_log_likelihood(scene_image=scene_image, crops=crops, object_names=object_names, question_text=question_text, candidate_text=choice_text, length_normalize=length_normalize)
            scores.append(score)
            logger.debug(f'  Choice {choice_idx}: score={score:.4f} | {choice_text[:60]}...')
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return (best_idx, scores)
