"""Open-vocabulary object detection using GroundingDINO.

GroundingDINO detects objects in images given text prompts, enabling
zero-shot detection of manipulation-relevant objects without fixed categories.

Reference: https://github.com/IDEA-Research/GroundingDINO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Default text prompts for manipulation-relevant objects
DEFAULT_OBJECT_PROMPTS = [
    "cup", "mug", "bottle", "bowl", "plate", "spoon", "fork", "knife",
    "tool", "screwdriver", "hammer", "pen", "pencil", "scissors",
    "phone", "remote", "key", "box", "container", "lid", "handle",
    "cloth", "towel", "sponge", "brush", "button", "switch", "knob",
    "drawer", "door handle", "bag", "book",
]


@dataclass
class DetectedObject:
    """A single detected object in a frame."""
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1
    center: list[float]  # [cx, cy] normalized 0-1
    area: float  # bbox area as fraction of frame


@dataclass
class FrameObjects:
    """Object detection results for a single frame."""
    frame_idx: int
    timestamp: float
    objects: list[DetectedObject] = field(default_factory=list)


def _load_grounding_dino(device: torch.device, model_path: str | None = None):
    """Load GroundingDINO model.

    Tries official groundingdino package, then falls back to HuggingFace
    transformers GroundingDINO, then to OWL-ViT as last resort.
    """
    # Try official GroundingDINO
    try:
        from groundingdino.util.inference import load_model, predict
        if model_path is None:
            # Try default config path
            import groundingdino
            pkg_dir = Path(groundingdino.__file__).parent
            config_path = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
            model = load_model(str(config_path), model_path or "groundingdino_swint_ogc.pth")
        else:
            model = load_model(model_path.replace(".pth", ".py"), model_path)
        model.to(device).eval()
        return model, "groundingdino", predict
    except (ImportError, FileNotFoundError):
        pass

    # Try HuggingFace transformers
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model.to(device).eval()
        return (model, processor), "transformers", None
    except (ImportError, OSError):
        pass

    # Fallback: OWL-ViT via transformers
    try:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        model.to(device).eval()
        return (model, processor), "owlvit", None
    except (ImportError, OSError):
        pass

    print("No object detection model available. Install groundingdino or transformers.")
    return None, "none", None


def _detect_groundingdino(
    frame_bgr: np.ndarray,
    model,
    predict_fn,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
) -> list[DetectedObject]:
    """Run official GroundingDINO."""
    from groundingdino.util.inference import load_image
    import torchvision.transforms.functional as F

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Convert to tensor
    image_tensor = F.to_tensor(rgb).to(device)
    image_tensor = F.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    boxes, logits, phrases = predict_fn(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    results = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        box = box.cpu().numpy()
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        results.append(DetectedObject(
            label=phrase,
            confidence=round(float(logit), 4),
            bbox=[round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            center=[round(cx, 6), round(cy, 6)],
            area=round(area, 6),
        ))

    return results


def _detect_transformers(
    frame_bgr: np.ndarray,
    model_and_processor: tuple,
    text_prompt: str,
    box_threshold: float,
    device: torch.device,
) -> list[DetectedObject]:
    """Run HuggingFace GroundingDINO or OWL-ViT."""
    model, processor = model_and_processor
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    from PIL import Image
    pil_image = Image.fromarray(rgb)

    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    if hasattr(outputs, "pred_boxes"):
        # GroundingDINO transformers output
        target_sizes = torch.tensor([[h, w]], device=device)
        results_raw = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            target_sizes=target_sizes,
        )
    else:
        target_sizes = torch.tensor([[h, w]], device=device)
        results_raw = processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=target_sizes,
        )

    results = []
    if results_raw:
        r = results_raw[0]
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        labels = r.get("labels", r.get("text", ["object"] * len(scores)))

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            # Normalize to 0-1
            x1_n, y1_n = x1 / w, y1 / h
            x2_n, y2_n = x2 / w, y2 / h
            cx = (x1_n + x2_n) / 2
            cy = (y1_n + y2_n) / 2
            area = (x2_n - x1_n) * (y2_n - y1_n)

            label_str = label if isinstance(label, str) else f"object_{label}"

            results.append(DetectedObject(
                label=label_str,
                confidence=round(float(score), 4),
                bbox=[round(float(x1_n), 6), round(float(y1_n), 6),
                      round(float(x2_n), 6), round(float(y2_n), 6)],
                center=[round(float(cx), 6), round(float(cy), 6)],
                area=round(float(area), 6),
            ))

    return results


def detect_objects(
    video_path: str | Path,
    text_prompts: list[str] | None = None,
    sample_rate: int = 30,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    model_path: str | None = None,
    output_dir: str | Path = "data/objects",
) -> list[FrameObjects]:
    """Detect objects in video frames using open-vocabulary detection.

    Args:
        video_path: Path to the video file.
        text_prompts: List of object categories to detect. Uses defaults if None.
        sample_rate: Process every Nth frame.
        box_threshold: Minimum confidence for detected boxes.
        text_threshold: Minimum confidence for text matching.
        model_path: Optional path to model weights.
        output_dir: Directory for saving results.

    Returns:
        List of FrameObjects results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if text_prompts is None:
        text_prompts = DEFAULT_OBJECT_PROMPTS

    # Build text prompt string
    text_prompt = " . ".join(text_prompts) + " ."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backend, predict_fn = _load_grounding_dino(device, model_path)

    if backend == "none":
        print("Skipping object detection: no model available")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_objects: list[FrameObjects] = []

    try:
        for frame_idx in tqdm(range(total_frames), desc="Detecting objects", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                continue

            timestamp = round(frame_idx / fps, 4)

            if backend == "groundingdino":
                objects = _detect_groundingdino(
                    frame, model, predict_fn, text_prompt,
                    box_threshold, text_threshold, device,
                )
            elif backend in ("transformers", "owlvit"):
                objects = _detect_transformers(
                    frame, model, text_prompt, box_threshold, device,
                )
            else:
                objects = []

            frame_objects.append(FrameObjects(
                frame_idx=frame_idx,
                timestamp=timestamp,
                objects=objects,
            ))
    finally:
        cap.release()

    return frame_objects
