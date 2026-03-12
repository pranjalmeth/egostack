"""Object and hand segmentation using SAM2 (Segment Anything Model 2).

SAM2 provides high-quality segmentation masks for objects and hands, with
video-level tracking capability for consistent object identity across frames.

Reference: https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class SegmentationMask:
    """A single segmentation mask for an object in a frame."""
    object_id: int
    label: str  # e.g., "hand_left", "cup", "unknown"
    mask: np.ndarray  # H x W bool array
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1
    area_fraction: float  # fraction of frame covered by mask


@dataclass
class FrameSegmentation:
    """Segmentation results for a single frame."""
    frame_idx: int
    timestamp: float
    masks: list[SegmentationMask] = field(default_factory=list)


def _load_sam2(device: torch.device, model_path: str | None = None, model_size: str = "small"):
    """Load SAM2 model.

    Tries sam2 package first, then falls back to the original SAM.
    """
    # Try SAM2
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_cfg_map = {
            "tiny": "sam2_hiera_t.yaml",
            "small": "sam2_hiera_s.yaml",
            "base_plus": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
        }
        cfg = model_cfg_map.get(model_size, "sam2_hiera_s.yaml")

        if model_path:
            sam2_model = build_sam2(cfg, model_path, device=device)
        else:
            sam2_model = build_sam2(cfg, device=device)

        predictor = SAM2ImagePredictor(sam2_model)
        return predictor, "sam2"
    except (ImportError, FileNotFoundError, RuntimeError):
        pass

    # Try original SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor

        model_type = "vit_b"
        if model_path:
            sam = sam_model_registry[model_type](checkpoint=model_path)
        else:
            sam = sam_model_registry[model_type]()
        sam.to(device)
        predictor = SamPredictor(sam)
        return predictor, "sam"
    except (ImportError, FileNotFoundError, RuntimeError):
        pass

    print("No SAM model available. Install sam2 or segment_anything.")
    return None, "none"


def _masks_from_points(
    predictor,
    image_rgb: np.ndarray,
    points: list[list[float]],
    labels: list[int],
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM/SAM2 with point prompts."""
    predictor.set_image(image_rgb)

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def _masks_from_boxes(
    predictor,
    image_rgb: np.ndarray,
    boxes: list[list[float]],
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SAM/SAM2 with box prompts."""
    predictor.set_image(image_rgb)

    input_boxes = np.array(boxes, dtype=np.float32)

    all_masks = []
    all_scores = []

    for box in input_boxes:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )
        all_masks.append(masks[0])
        all_scores.append(scores[0])

    return np.array(all_masks), np.array(all_scores)


def segment_frame(
    frame_bgr: np.ndarray,
    predictor,
    backend: str,
    hand_bboxes: list[dict] | None = None,
    object_bboxes: list[dict] | None = None,
    point_prompts: list[dict] | None = None,
) -> list[SegmentationMask]:
    """Segment a single frame given bounding box or point prompts.

    Args:
        frame_bgr: BGR image.
        predictor: SAM/SAM2 predictor.
        backend: "sam2" or "sam".
        hand_bboxes: List of hand bounding boxes [{label, bbox: [x1,y1,x2,y2]}].
        object_bboxes: List of object bounding boxes [{label, bbox: [x1,y1,x2,y2], confidence}].
        point_prompts: List of point prompts [{label, points: [[x,y]], labels: [0/1]}].

    Returns:
        List of SegmentationMask results.
    """
    if predictor is None:
        return []

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    results = []
    obj_id = 0

    # Process hand bounding boxes
    if hand_bboxes:
        boxes_pixel = []
        box_labels = []
        for hb in hand_bboxes:
            bbox = hb["bbox"]
            # Convert normalized to pixel coordinates
            boxes_pixel.append([
                bbox[0] * w, bbox[1] * h,
                bbox[2] * w, bbox[3] * h,
            ])
            box_labels.append(hb.get("label", "hand"))

        if boxes_pixel:
            masks, scores = _masks_from_boxes(predictor, rgb, boxes_pixel, backend)

            for mask, score, label in zip(masks, scores, box_labels):
                area_frac = float(mask.sum()) / (h * w)
                # Compute mask bbox
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    mbbox = [
                        round(float(xs.min()) / w, 6),
                        round(float(ys.min()) / h, 6),
                        round(float(xs.max()) / w, 6),
                        round(float(ys.max()) / h, 6),
                    ]
                else:
                    mbbox = [0.0, 0.0, 0.0, 0.0]

                results.append(SegmentationMask(
                    object_id=obj_id,
                    label=label,
                    mask=mask.astype(bool),
                    confidence=round(float(score), 4),
                    bbox=mbbox,
                    area_fraction=round(area_frac, 6),
                ))
                obj_id += 1

    # Process object bounding boxes
    if object_bboxes:
        boxes_pixel = []
        box_labels = []
        box_confs = []
        for ob in object_bboxes:
            bbox = ob["bbox"]
            boxes_pixel.append([
                bbox[0] * w, bbox[1] * h,
                bbox[2] * w, bbox[3] * h,
            ])
            box_labels.append(ob.get("label", "object"))
            box_confs.append(ob.get("confidence", 0.5))

        if boxes_pixel:
            masks, scores = _masks_from_boxes(predictor, rgb, boxes_pixel, backend)

            for mask, score, label in zip(masks, scores, box_labels):
                area_frac = float(mask.sum()) / (h * w)
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    mbbox = [
                        round(float(xs.min()) / w, 6),
                        round(float(ys.min()) / h, 6),
                        round(float(xs.max()) / w, 6),
                        round(float(ys.max()) / h, 6),
                    ]
                else:
                    mbbox = [0.0, 0.0, 0.0, 0.0]

                results.append(SegmentationMask(
                    object_id=obj_id,
                    label=label,
                    mask=mask.astype(bool),
                    confidence=round(float(score), 4),
                    bbox=mbbox,
                    area_fraction=round(area_frac, 6),
                ))
                obj_id += 1

    # Process point prompts
    if point_prompts:
        for pp in point_prompts:
            pts = [[p[0] * w, p[1] * h] for p in pp["points"]]
            lbls = pp.get("labels", [1] * len(pts))
            masks, scores, _ = _masks_from_points(predictor, rgb, pts, lbls, backend)

            # Use the highest-scoring mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]

            area_frac = float(mask.sum()) / (h * w)
            ys, xs = np.where(mask)
            if len(xs) > 0:
                mbbox = [
                    round(float(xs.min()) / w, 6),
                    round(float(ys.min()) / h, 6),
                    round(float(xs.max()) / w, 6),
                    round(float(ys.max()) / h, 6),
                ]
            else:
                mbbox = [0.0, 0.0, 0.0, 0.0]

            results.append(SegmentationMask(
                object_id=obj_id,
                label=pp.get("label", "object"),
                mask=mask.astype(bool),
                confidence=round(float(score), 4),
                bbox=mbbox,
                area_fraction=round(area_frac, 6),
            ))
            obj_id += 1

    return results


def segment_video(
    video_path: str | Path,
    hand_detections: list | None = None,
    object_detections: list | None = None,
    sample_rate: int = 1,
    model_path: str | None = None,
    model_size: str = "small",
    output_dir: str | Path = "data/segmentation",
    save_masks: bool = True,
) -> list[FrameSegmentation]:
    """Segment hands and objects across video frames.

    Args:
        video_path: Path to the video file.
        hand_detections: Per-frame hand detections (FrameDetection list).
        object_detections: Per-frame object detections (FrameObjects list).
        sample_rate: Process every Nth frame.
        model_path: Optional path to SAM2 checkpoint.
        model_size: SAM2 model size ("tiny", "small", "base_plus", "large").
        output_dir: Directory for saving mask outputs.
        save_masks: Whether to save mask arrays to disk.

    Returns:
        List of FrameSegmentation results.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    mask_dir = output_dir / video_path.stem
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor, backend = _load_sam2(device, model_path, model_size)

    if backend == "none":
        print("Skipping segmentation: no model available")
        return []

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

    frame_segmentations: list[FrameSegmentation] = []

    try:
        for frame_idx in tqdm(range(total_frames), desc="Segmenting frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                continue

            timestamp = round(frame_idx / fps, 4)

            # Build prompts from detections
            hand_bboxes = []
            hdet = hand_det_by_frame.get(frame_idx)
            if hdet:
                hands = hdet.hands if hasattr(hdet, "hands") else []
                for h in hands:
                    landmarks = h.landmarks if hasattr(h, "landmarks") else h.get("landmarks", [])
                    if not landmarks:
                        continue
                    xs = [lm["x"] for lm in landmarks]
                    ys = [lm["y"] for lm in landmarks]
                    pad = 0.05
                    hand_bboxes.append({
                        "label": f"hand_{h.hand_label if hasattr(h, 'hand_label') else h.get('hand_label', 'unknown')}",
                        "bbox": [
                            max(0, min(xs) - pad),
                            max(0, min(ys) - pad),
                            min(1, max(xs) + pad),
                            min(1, max(ys) + pad),
                        ],
                    })

            object_bboxes = []
            odet = obj_det_by_frame.get(frame_idx)
            if odet:
                objects = odet.objects if hasattr(odet, "objects") else []
                for obj in objects:
                    object_bboxes.append({
                        "label": obj.label if hasattr(obj, "label") else obj.get("label", "object"),
                        "bbox": obj.bbox if hasattr(obj, "bbox") else obj.get("bbox", [0, 0, 1, 1]),
                        "confidence": obj.confidence if hasattr(obj, "confidence") else obj.get("confidence", 0.5),
                    })

            masks = segment_frame(
                frame, predictor, backend,
                hand_bboxes=hand_bboxes or None,
                object_bboxes=object_bboxes or None,
            )

            # Save masks to disk
            if save_masks and masks:
                mask_arrays = {}
                for m in masks:
                    mask_arrays[f"mask_{m.object_id}_{m.label}"] = m.mask.astype(np.uint8)
                npz_path = mask_dir / f"frame_{frame_idx:06d}_masks.npz"
                np.savez_compressed(str(npz_path), **mask_arrays)

            frame_segmentations.append(FrameSegmentation(
                frame_idx=frame_idx,
                timestamp=timestamp,
                masks=masks,
            ))
    finally:
        cap.release()

    return frame_segmentations
