from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from src.dataset import BoundingBox, EvalResult, VCRSample, detokenize_with_tags, load_metadata, load_vcr_dataset, load_vcr_from_hf
from src.hf_dataloader import VCRStreamDataset, VCRItem
from src.perception import PerceptionLayer
from src.cognition import CognitionLayer
logging.basicConfig(level=logging.INFO, format='%(asctime)s │ %(levelname)-7s │ %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def configure_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('✓ MPS backend detected — using Apple Silicon GPU acceleration')
    else:
        device = torch.device('cpu')
        logger.warning('✗ MPS backend NOT available. Falling back to CPU. Ensure you are running native ARM64 Python with PyTorch ≥ 2.1.')
    return device

def evaluate_vcr(dataset: list[VCRSample], perception: PerceptionLayer, cognition: CognitionLayer, length_normalize: bool=True) -> list[EvalResult]:
    results: list[EvalResult] = []
    progress_bar = tqdm(dataset, desc='Evaluating VCR', unit='sample')
    for sample in progress_bar:
        try:
            scene_image = sample.image.convert('RGB') if sample.image is not None else Image.open(sample.img_path).convert('RGB')
            boxes = sample.boxes if sample.boxes is not None else load_metadata(sample.metadata_path)
            crops = perception.crop_from_boxes(scene_image, boxes, sample.objects)
            question_text = sample.question_text or detokenize_with_tags(sample.question, sample.objects)
            answer_texts: list[str] = sample.answer_choice_texts or [detokenize_with_tags(choice, sample.objects) for choice in sample.answer_choices]
            rationale_texts: list[str] = sample.rationale_choice_texts or [detokenize_with_tags(choice, sample.objects) for choice in sample.rationale_choices]
            logger.info(f'Sample {sample.index}: Q→A scoring...')
            predicted_answer, answer_scores = cognition.score_multiple_choice(scene_image=scene_image, crops=crops, object_names=sample.objects, question_text=question_text, choices=answer_texts, length_normalize=length_normalize)
            conditioned_question = f'{question_text}\nThe correct answer is: "{answer_texts[predicted_answer]}".\nWhat is the most plausible visual reasoning or rationale that explains why this answer is correct based on what you see in the images?'
            logger.info(f'Sample {sample.index}: Q→R scoring (conditioned on predicted A={predicted_answer})...')
            predicted_rationale, rationale_scores = cognition.score_multiple_choice(scene_image=scene_image, crops=crops, object_names=sample.objects, question_text=conditioned_question, choices=rationale_texts, length_normalize=length_normalize)
            answer_correct = predicted_answer == sample.answer_label
            rationale_correct = predicted_rationale == sample.rationale_label
            joint_correct = answer_correct and rationale_correct
            result = EvalResult(sample_index=sample.index, predicted_answer=predicted_answer, predicted_rationale=predicted_rationale, gt_answer=sample.answer_label, gt_rationale=sample.rationale_label, answer_correct=answer_correct, rationale_correct=rationale_correct, joint_correct=joint_correct, answer_log_likelihoods=answer_scores, rationale_log_likelihoods=rationale_scores)
            results.append(result)
            running_qa = sum((r.answer_correct for r in results)) / len(results)
            running_qar = sum((r.joint_correct for r in results)) / len(results)
            progress_bar.set_postfix({'Q→A': f'{running_qa:.1%}', 'Q→AR': f'{running_qar:.1%}', '✓' if joint_correct else '✗': f'A={predicted_answer}/R={predicted_rationale}'})
            logger.info(f"  Sample {sample.index}: A={predicted_answer}(GT={sample.answer_label}) {('✓' if answer_correct else '✗')} | R={predicted_rationale}(GT={sample.rationale_label}) {('✓' if rationale_correct else '✗')} | Q→AR={('✓' if joint_correct else '✗')}")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f'Error processing sample {sample.index}: {e}', exc_info=True)
            results.append(EvalResult(sample_index=sample.index, predicted_answer=-1, predicted_rationale=-1, gt_answer=sample.answer_label, gt_rationale=sample.rationale_label, answer_correct=False, rationale_correct=False, joint_correct=False))
    return results

def compute_metrics(results: list[EvalResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {'Q→A': 0.0, 'Q→R': 0.0, 'Q→AR': 0.0, 'num_samples': 0}
    qa_correct = sum((r.answer_correct for r in results))
    qr_correct = sum((r.rationale_correct for r in results))
    qar_correct = sum((r.joint_correct for r in results))
    return {'Q→A': qa_correct / n, 'Q→R': qr_correct / n, 'Q→AR': qar_correct / n, 'num_samples': n, 'num_correct_A': qa_correct, 'num_correct_R': qr_correct, 'num_correct_AR': qar_correct}

def main() -> None:
    parser = argparse.ArgumentParser(description='VCR Evaluation Pipeline — Hybrid Neuro-Symbolic System', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python -m src.evaluate --vcr-root ./vcr1 --split val --num-samples 5\n  python -m src.evaluate --vcr-root ./vcr1 --split val\n  python -m src.evaluate --vcr-root ./vcr1 --split val \\\n      --qwen-model Qwen/Qwen2.5-VL-7B-Instruct \\\n      --yolo-model yolo11n.pt\n        ')
    parser.add_argument('--vcr-root', type=Path, default=None, help='Root directory of the VCR dataset (containing vcr1images/ and *.jsonl). Required unless --use-hf-stream is set.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Which dataset split to evaluate (default: val)')
    parser.add_argument('--num-samples', type=int, default=None, help='Evaluate only the first N samples (for debugging). Default: all.')
    parser.add_argument('--qwen-model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='Hugging Face model ID for Qwen2.5-VL (default: Qwen/Qwen2.5-VL-7B-Instruct)')
    parser.add_argument('--yolo-model', type=str, default='yolo11n.pt', help='YOLO model weights file (default: yolo11n.pt)')
    parser.add_argument('--no-length-norm', action='store_true', help='Disable length normalization of log-likelihoods.')
    parser.add_argument('--use-hf-stream', action='store_true', help='Stream images + annotations from HuggingFace (no local files needed).')
    parser.add_argument('--use-hf-dataset', action='store_true', help='Load VCR annotations from HuggingFace Hub (Rowan/vcr) instead of local JSONL files. Still requires --vcr-root for images.')
    parser.add_argument('--output', type=Path, default=None, help='Path to save JSON results file. Default: vcr_results_<split>.json')
    args = parser.parse_args()
    if args.use_hf_stream:
        device = configure_device()
        logger.info('=' * 70)
        logger.info('VCR EVALUATION PIPELINE — Configuration')
        logger.info('=' * 70)
        logger.info(f'  Data source:      HuggingFace Streaming (Rowan/vcr)')
        logger.info(f'  Split:            {args.split}')
        logger.info(f"  Num samples:      {args.num_samples or 'ALL'}")
        logger.info(f'  Qwen model:       {args.qwen_model}')
        logger.info(f'  YOLO model:       {args.yolo_model}')
        logger.info(f'  Device:           {device}')
        logger.info(f'  Length normalize:  {not args.no_length_norm}')
        logger.info('=' * 70)
        logger.info('Initializing perception layer (YOLO)...')
        perception = PerceptionLayer(yolo_model_name=args.yolo_model, device=device)
        logger.info('Initializing cognition layer (Qwen2.5-VL)...')
        cognition = CognitionLayer(model_name=args.qwen_model, device=device)
        logger.info('Streaming VCR dataset from HuggingFace...')
        stream_dataset = VCRStreamDataset(split=args.split, max_samples=args.num_samples)
        samples: list[VCRSample] = []
        for item in stream_dataset:
            samples.append(VCRSample(index=item.sample_index, img_path=Path(item.img_fn), metadata_path=Path(item.img_fn), objects=item.objects, question=[], answer_choices=[], rationale_choices=[], answer_label=item.answer_label, rationale_label=item.rationale_label, question_text=item.question_text, answer_choice_texts=item.answer_choices, rationale_choice_texts=item.rationale_choices, image=item.image, boxes=item.boxes))
        if not samples:
            logger.error('No valid samples streamed. Check your connection.')
            sys.exit(1)
        logger.info(f'Streamed {len(samples)} samples. Starting evaluation...')
        start_time = time.time()
        results = evaluate_vcr(dataset=samples, perception=perception, cognition=cognition, length_normalize=not args.no_length_norm)
        elapsed = time.time() - start_time
        metrics = compute_metrics(results)
        logger.info('')
        logger.info('═' * 70)
        logger.info('                    EVALUATION RESULTS')
        logger.info('═' * 70)
        logger.info(f"  Total samples:  {metrics['num_samples']}")
        logger.info(f'  Time elapsed:   {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/sample)')
        logger.info('─' * 70)
        logger.info(f"  Q→A  Accuracy:  {metrics['Q→A']:.4f}  ({metrics['num_correct_A']}/{metrics['num_samples']})")
        logger.info(f"  Q→R  Accuracy:  {metrics['Q→R']:.4f}  ({metrics['num_correct_R']}/{metrics['num_samples']})")
        logger.info(f"  Q→AR Accuracy:  {metrics['Q→AR']:.4f}  ({metrics['num_correct_AR']}/{metrics['num_samples']})")
        logger.info('═' * 70)
        output_path = args.output or Path(f'vcr_results_{args.split}.json')
        output_data = {'config': {'data_source': 'HuggingFace Streaming (Rowan/vcr)', 'split': args.split, 'qwen_model': args.qwen_model, 'yolo_model': args.yolo_model, 'length_normalize': not args.no_length_norm, 'device': str(device), 'elapsed_seconds': elapsed}, 'metrics': metrics, 'per_sample_results': [{'sample_index': r.sample_index, 'predicted_answer': r.predicted_answer, 'predicted_rationale': r.predicted_rationale, 'gt_answer': r.gt_answer, 'gt_rationale': r.gt_rationale, 'answer_correct': r.answer_correct, 'rationale_correct': r.rationale_correct, 'joint_correct': r.joint_correct, 'answer_log_likelihoods': r.answer_log_likelihoods, 'rationale_log_likelihoods': r.rationale_log_likelihoods} for r in results]}
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f'Detailed results saved to: {output_path}')
        return
    if args.vcr_root is None:
        logger.error('--vcr-root is required unless --use-hf-stream is set.')
        sys.exit(1)
    vcr_root = args.vcr_root.resolve()
    images_root = vcr_root / 'vcr1images'
    if not vcr_root.exists():
        logger.error(f'VCR root directory not found: {vcr_root}')
        sys.exit(1)
    if not images_root.exists():
        logger.error(f'Images directory not found: {images_root}')
        sys.exit(1)
    if not args.use_hf_dataset:
        jsonl_path = vcr_root / f'{args.split}.jsonl'
        if not jsonl_path.exists():
            logger.error(f'Annotation file not found: {jsonl_path}')
            sys.exit(1)
    device = configure_device()
    data_source = 'HuggingFace (Rowan/vcr)' if args.use_hf_dataset else 'Local JSONL'
    logger.info('=' * 70)
    logger.info('VCR EVALUATION PIPELINE — Configuration')
    logger.info('=' * 70)
    logger.info(f'  Data source:      {data_source}')
    logger.info(f'  Dataset root:     {vcr_root}')
    logger.info(f'  Split:            {args.split}')
    logger.info(f"  Num samples:      {args.num_samples or 'ALL'}")
    logger.info(f'  Qwen model:       {args.qwen_model}')
    logger.info(f'  YOLO model:       {args.yolo_model}')
    logger.info(f'  Device:           {device}')
    logger.info(f'  Length normalize:  {not args.no_length_norm}')
    logger.info('=' * 70)
    logger.info('Loading VCR dataset...')
    if args.use_hf_dataset:
        dataset = load_vcr_from_hf(split=args.split, images_root=images_root, num_samples=args.num_samples)
    else:
        dataset = load_vcr_dataset(jsonl_path, images_root, num_samples=args.num_samples)
    if not dataset:
        logger.error('No valid samples loaded. Check your dataset paths.')
        sys.exit(1)
    logger.info('Initializing perception layer (YOLO)...')
    perception = PerceptionLayer(yolo_model_name=args.yolo_model, device=device)
    logger.info('Initializing cognition layer (Qwen2.5-VL)...')
    cognition = CognitionLayer(model_name=args.qwen_model, device=device)
    logger.info('Starting evaluation...')
    start_time = time.time()
    results = evaluate_vcr(dataset=dataset, perception=perception, cognition=cognition, length_normalize=not args.no_length_norm)
    elapsed = time.time() - start_time
    metrics = compute_metrics(results)
    logger.info('')
    logger.info('═' * 70)
    logger.info('                    EVALUATION RESULTS')
    logger.info('═' * 70)
    logger.info(f"  Total samples:  {metrics['num_samples']}")
    logger.info(f'  Time elapsed:   {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/sample)')
    logger.info('─' * 70)
    logger.info(f"  Q→A  Accuracy:  {metrics['Q→A']:.4f}  ({metrics['num_correct_A']}/{metrics['num_samples']})")
    logger.info(f"  Q→R  Accuracy:  {metrics['Q→R']:.4f}  ({metrics['num_correct_R']}/{metrics['num_samples']})")
    logger.info(f"  Q→AR Accuracy:  {metrics['Q→AR']:.4f}  ({metrics['num_correct_AR']}/{metrics['num_samples']})")
    logger.info('═' * 70)
    output_path = args.output or Path(f'vcr_results_{args.split}.json')
    output_data = {'config': {'vcr_root': str(vcr_root), 'data_source': data_source, 'split': args.split, 'qwen_model': args.qwen_model, 'yolo_model': args.yolo_model, 'length_normalize': not args.no_length_norm, 'device': str(device), 'elapsed_seconds': elapsed}, 'metrics': metrics, 'per_sample_results': [{'sample_index': r.sample_index, 'predicted_answer': r.predicted_answer, 'predicted_rationale': r.predicted_rationale, 'gt_answer': r.gt_answer, 'gt_rationale': r.gt_rationale, 'answer_correct': r.answer_correct, 'rationale_correct': r.rationale_correct, 'joint_correct': r.joint_correct, 'answer_log_likelihoods': r.answer_log_likelihoods, 'rationale_log_likelihoods': r.rationale_log_likelihoods} for r in results]}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f'Detailed results saved to: {output_path}')
if __name__ == '__main__':
    main()
