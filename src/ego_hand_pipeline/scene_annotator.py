"""Scene annotation using Vision-Language Models (VLMs).

Generates natural language descriptions of what's happening in each scene,
combining visual understanding with detected hands, objects, and actions.
Falls back to rule-based descriptions if no VLM is available.

Supported backends:
1. Transformers BLIP-2 / InstructBLIP (local GPU)
2. Rule-based descriptions from detection data (no model needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class SceneAnnotation:
    """Scene description for a single frame or clip segment."""
    frame_idx: int
    timestamp: float
    caption: str
    objects_present: list[str] = field(default_factory=list)
    hands_present: list[str] = field(default_factory=list)
    action_description: str = ""
    confidence: float = 0.0


def _load_vlm(device, model_name: str = "blip2"):
    """Load a Vision-Language Model for scene captioning.

    Tries BLIP-2, then InstructBLIP, then falls back to rule-based.
    """
    import torch

    if model_name == "blip2":
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16,
            )
            model.to(device).eval()
            return (model, processor), "blip2"
        except (ImportError, OSError):
            pass

    if model_name in ("blip2", "instructblip"):
        try:
            from transformers import (
                InstructBlipProcessor,
                InstructBlipForConditionalGeneration,
            )
            processor = InstructBlipProcessor.from_pretrained(
                "Salesforce/instructblip-vicuna-7b"
            )
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b",
                torch_dtype=torch.float16,
            )
            model.to(device).eval()
            return (model, processor), "instructblip"
        except (ImportError, OSError):
            pass

    # Fallback to rule-based
    return None, "rule_based"


def _caption_with_vlm(
    frame_rgb: np.ndarray,
    model_and_processor: tuple,
    backend: str,
    prompt: str = "Describe what the person's hands are doing in this egocentric view.",
) -> str:
    """Generate a caption using a VLM."""
    import torch
    from PIL import Image

    model, processor = model_and_processor
    pil_image = Image.fromarray(frame_rgb)

    if backend == "blip2":
        inputs = processor(images=pil_image, return_tensors="pt").to(
            model.device, torch.float16
        )
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
    elif backend == "instructblip":
        inputs = processor(
            images=pil_image, text=prompt, return_tensors="pt"
        ).to(model.device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
    else:
        caption = ""

    return caption


def _generate_rule_based_caption(
    hands_info: list[dict],
    objects_info: list[dict],
) -> str:
    """Generate a scene description from detection data without a VLM.

    Uses detected hands and objects to construct a natural language description.
    """
    parts = []

    # Describe hands
    hand_labels = [h.get("label", "hand") for h in hands_info]
    if len(hand_labels) == 2:
        parts.append("Both hands are visible")
    elif len(hand_labels) == 1:
        parts.append(f"The {hand_labels[0]} hand is visible")
    else:
        parts.append("No hands detected")

    # Describe objects
    obj_labels = [o.get("label", "object") for o in objects_info]
    unique_objects = list(dict.fromkeys(obj_labels))  # dedupe preserving order

    if unique_objects:
        if len(unique_objects) == 1:
            parts.append(f"near a {unique_objects[0]}")
        elif len(unique_objects) == 2:
            parts.append(f"near a {unique_objects[0]} and {unique_objects[1]}")
        else:
            obj_str = ", ".join(unique_objects[:-1]) + f", and {unique_objects[-1]}"
            parts.append(f"near {obj_str}")

    # Infer action from spatial relationships
    if hands_info and objects_info:
        # Check if any hand bbox overlaps with an object bbox
        for hand in hands_info:
            hand_bbox = hand.get("bbox")
            if not hand_bbox:
                continue
            for obj in objects_info:
                obj_bbox = obj.get("bbox")
                if not obj_bbox:
                    continue
                if _bboxes_overlap(hand_bbox, obj_bbox):
                    parts.append(
                        f"— the {hand.get('label', 'hand')} hand appears to be "
                        f"interacting with the {obj.get('label', 'object')}"
                    )
                    break

    caption = " ".join(parts) + "."
    return caption


def _bboxes_overlap(bbox_a: list[float], bbox_b: list[float]) -> bool:
    """Check if two normalized [x1,y1,x2,y2] bounding boxes overlap."""
    return not (
        bbox_a[2] < bbox_b[0] or bbox_b[2] < bbox_a[0] or
        bbox_a[3] < bbox_b[1] or bbox_b[3] < bbox_a[1]
    )


def annotate_scenes(
    video_path: str | Path,
    hand_detections: list | None = None,
    object_detections: list | None = None,
    sample_rate: int = 30,
    model_name: str = "blip2",
    output_dir: str | Path = "data/scene_annotations",
) -> list[SceneAnnotation]:
    """Generate scene annotations for video frames.

    Args:
        video_path: Path to the video file.
        hand_detections: Per-frame hand detections.
        object_detections: Per-frame object detections.
        sample_rate: Annotate every Nth frame.
        model_name: VLM to use ("blip2", "instructblip", "rule_based").
        output_dir: Directory for saving annotations.

    Returns:
        List of SceneAnnotation results.
    """
    import torch

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VLM (or fall back to rule-based)
    if model_name == "rule_based":
        vlm, backend = None, "rule_based"
    else:
        vlm, backend = _load_vlm(device, model_name)

    # Build detection lookups
    hand_det_by_frame = {}
    if hand_detections:
        for det in hand_detections:
            fidx = det.frame_idx if hasattr(det, "frame_idx") else det.get("frame_idx", -1)
            hand_det_by_frame[fidx] = det

    obj_det_by_frame = {}
    if object_detections:
        for det in object_detections:
            fidx = det.frame_idx if hasattr(det, "frame_idx") else det.get("frame_idx", -1)
            obj_det_by_frame[fidx] = det

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotations: list[SceneAnnotation] = []

    try:
        for frame_idx in tqdm(range(total_frames), desc="Annotating scenes",
                              unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                continue

            timestamp = round(frame_idx / fps, 4)

            # Gather hand info
            hands_info = []
            hdet = hand_det_by_frame.get(frame_idx)
            if hdet:
                hands = hdet.hands if hasattr(hdet, "hands") else []
                for h in hands:
                    label = h.hand_label if hasattr(h, "hand_label") else h.get("hand_label", "unknown")
                    landmarks = h.landmarks if hasattr(h, "landmarks") else h.get("landmarks", [])
                    bbox = None
                    if landmarks:
                        xs = [lm["x"] for lm in landmarks]
                        ys = [lm["y"] for lm in landmarks]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    hands_info.append({"label": label, "bbox": bbox})

            # Gather object info
            objects_info = []
            odet = obj_det_by_frame.get(frame_idx)
            if odet:
                objects = odet.objects if hasattr(odet, "objects") else []
                for obj in objects:
                    label = obj.label if hasattr(obj, "label") else obj.get("label", "object")
                    bbox = obj.bbox if hasattr(obj, "bbox") else obj.get("bbox")
                    objects_info.append({"label": label, "bbox": bbox})

            # Generate caption
            if backend != "rule_based" and vlm is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                caption = _caption_with_vlm(rgb, vlm, backend)
                confidence = 0.8
            else:
                caption = _generate_rule_based_caption(hands_info, objects_info)
                confidence = 0.5

            hand_labels = [h["label"] for h in hands_info]
            obj_labels = [o["label"] for o in objects_info]

            annotations.append(SceneAnnotation(
                frame_idx=frame_idx,
                timestamp=timestamp,
                caption=caption,
                objects_present=obj_labels,
                hands_present=hand_labels,
                action_description=caption,
                confidence=confidence,
            ))
    finally:
        cap.release()

    return annotations
