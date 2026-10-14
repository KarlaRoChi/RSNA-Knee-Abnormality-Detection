import numpy as np
import pytest

from rsna_knee.metrics import macro_auc


def test_macro_auc_ignores_single_class_targets():
    y_true = np.array([[0, 0], [1, 0], [1, 0]])
    y_score = np.array([[0.1, 0.3], [0.8, 0.4], [0.9, 0.5]])
    assert macro_auc(y_true, y_score) == pytest.approx(1.0)


def test_macro_auc_requires_a_valid_target():
    with pytest.raises(ValueError, match="undefined"):
        macro_auc(np.zeros((2, 1)), np.zeros((2, 1)))
