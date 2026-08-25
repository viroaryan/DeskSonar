"""
Unit Tests for DeskSonar Real-Time ML Gesture Neural Network
"""
import numpy as np
import pytest
from src.ai.gesture_classifier_net import AcousticGestureNet, AcousticMLManager, GESTURE_CLASSES


def test_gesture_net_forward():
    net = AcousticGestureNet()
    X_spec = np.random.randn(4, 32, 32).astype(np.float32)
    X_phase = np.random.randn(4, 8).astype(np.float32)

    probs = net.forward(X_spec, X_phase)
    assert probs.shape == (4, len(GESTURE_CLASSES))
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)


def test_gesture_net_training_step():
    net = AcousticGestureNet()
    X_spec = np.random.randn(8, 32, 32).astype(np.float32)
    X_phase = np.random.randn(8, 8).astype(np.float32)
    y = np.random.randint(0, len(GESTURE_CLASSES), size=8)

    initial_loss = net.train_step(X_spec, X_phase, y, lr=0.01)
    assert initial_loss > 0.0

    # Loss should decrease after multiple steps on same data
    for _ in range(15):
        loss = net.train_step(X_spec, X_phase, y, lr=0.01)
    assert loss < initial_loss


def test_ml_manager_predict():
    ml = AcousticMLManager()
    spec = np.zeros((32, 32), dtype=np.float32)
    phase = np.zeros(8, dtype=np.float32)

    pred, conf, all_probs = ml.predict(spec, phase)
    assert pred in GESTURE_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(all_probs) == len(GESTURE_CLASSES)
