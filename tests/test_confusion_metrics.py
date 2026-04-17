"""Tests for accuracy metrics derived from confusion matrices."""

import numpy as np

from desert_segmentation.metrics.iou import (
    confusion_to_accuracy_metrics,
    valid_class_miou_from_confusion,
)


def test_perfect_confusion():
    cm = np.eye(3, dtype=np.int64) * 100
    out = confusion_to_accuracy_metrics(cm)
    assert out["global_pixel_accuracy"] == 1.0
    assert out["mean_class_accuracy"] == 1.0
    assert np.allclose(out["per_class_recall"], [1.0, 1.0, 1.0])


def test_two_class_mixed():
    # GT: 100 class0, 100 class1; half wrong each
    cm = np.array([[50, 50], [50, 50]], dtype=np.int64)
    out = confusion_to_accuracy_metrics(cm)
    assert out["global_pixel_accuracy"] == 0.5
    assert abs(out["mean_class_accuracy"] - 0.5) < 1e-6


def test_one_class_absent_in_gt():
    # Only class 0 appears in val GT; 80 correct, 20 predicted as class 1.
    cm = np.array([[80, 20], [0, 0]], dtype=np.int64)
    out = confusion_to_accuracy_metrics(cm)
    assert abs(out["global_pixel_accuracy"] - 0.8) < 1e-9
    assert abs(out["mean_class_accuracy"] - 0.8) < 1e-9
    assert np.isnan(out["per_class_recall"][1])


def test_valid_class_miou_only_classes_with_gt():
    # Two classes in GT; full mIoU averages zeros for empty rows if any — here both rows have GT.
    cm = np.array([[90, 10], [10, 90]], dtype=np.int64)
    # IoU class0: 90/(90+10+10)=90/110, class1: 90/110
    v = valid_class_miou_from_confusion(cm)
    iou0 = 90.0 / (90 + 10 + 10)
    assert abs(v - iou0) < 1e-6


def test_valid_class_miou_ignores_empty_gt_rows():
    # Class 1 has no GT pixels; valid-class mIoU averages only class 0.
    cm = np.array([[80, 20], [0, 0]], dtype=np.int64)
    v = valid_class_miou_from_confusion(cm)
    # IoU class 0: TP=80, union = rows[0]+cols[0]-TP = 100+80-80 = 100
    assert abs(v - 0.8) < 1e-6
