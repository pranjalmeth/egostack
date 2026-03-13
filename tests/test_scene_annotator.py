"""Tests for scene_annotator module."""

import pytest


def test_bboxes_overlap_true():
    """Overlapping bounding boxes should return True."""
    from ego_hand_pipeline.scene_annotator import _bboxes_overlap

    assert _bboxes_overlap([0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7])


def test_bboxes_overlap_false():
    """Non-overlapping bounding boxes should return False."""
    from ego_hand_pipeline.scene_annotator import _bboxes_overlap

    assert not _bboxes_overlap([0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7])


def test_bboxes_overlap_edge_touching():
    """Edge-touching bounding boxes should return True (shared boundary)."""
    from ego_hand_pipeline.scene_annotator import _bboxes_overlap

    assert _bboxes_overlap([0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5])


def test_rule_based_caption_both_hands_with_objects():
    """Rule-based caption with two hands and objects."""
    from ego_hand_pipeline.scene_annotator import _generate_rule_based_caption

    hands = [
        {"label": "left", "bbox": [0.1, 0.1, 0.3, 0.3]},
        {"label": "right", "bbox": [0.5, 0.1, 0.7, 0.3]},
    ]
    objects = [
        {"label": "cup", "bbox": [0.2, 0.2, 0.4, 0.4]},
    ]

    caption = _generate_rule_based_caption(hands, objects)
    assert "Both hands" in caption
    assert "cup" in caption


def test_rule_based_caption_one_hand():
    """Rule-based caption with one hand."""
    from ego_hand_pipeline.scene_annotator import _generate_rule_based_caption

    hands = [{"label": "right", "bbox": [0.5, 0.1, 0.7, 0.3]}]
    caption = _generate_rule_based_caption(hands, [])
    assert "right hand" in caption


def test_rule_based_caption_no_hands():
    """Rule-based caption with no hands detected."""
    from ego_hand_pipeline.scene_annotator import _generate_rule_based_caption

    caption = _generate_rule_based_caption([], [])
    assert "No hands" in caption


def test_rule_based_caption_interaction_detected():
    """Rule-based caption should detect hand-object interaction."""
    from ego_hand_pipeline.scene_annotator import _generate_rule_based_caption

    hands = [{"label": "right", "bbox": [0.3, 0.3, 0.6, 0.6]}]
    objects = [{"label": "bottle", "bbox": [0.4, 0.4, 0.7, 0.7]}]

    caption = _generate_rule_based_caption(hands, objects)
    assert "interacting" in caption
    assert "bottle" in caption


def test_rule_based_caption_multiple_objects():
    """Rule-based caption with multiple objects."""
    from ego_hand_pipeline.scene_annotator import _generate_rule_based_caption

    hands = [{"label": "left", "bbox": None}]
    objects = [
        {"label": "cup", "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"label": "spoon", "bbox": [0.3, 0.3, 0.4, 0.4]},
        {"label": "plate", "bbox": [0.5, 0.5, 0.6, 0.6]},
    ]

    caption = _generate_rule_based_caption(hands, objects)
    assert "cup" in caption
    assert "spoon" in caption
    assert "plate" in caption
