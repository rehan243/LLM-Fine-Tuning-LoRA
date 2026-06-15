import pytest
from training_utils import calculate_learning_rate, adjust_weights

def test_calculate_learning_rate():
    # testing learning rate calculation
    base_lr = 0.01
    epoch = 5
    expected_lr = base_lr * (0.95 ** epoch)  # assuming a simple decay
    assert calculate_learning_rate(base_lr, epoch) == expected_lr

def test_adjust_weights():
    # testing weight adjustment
    weights = [0.2, 0.5, 0.3]
    gradient = [0.1, 0.2, 0.1]
    learning_rate = 0.01
    expected_weights = [w - learning_rate * g for w, g in zip(weights, gradient)]
    
    adjusted_weights = adjust_weights(weights, gradient, learning_rate)
    assert adjusted_weights == expected_weights

def test_adjust_weights_no_change():
    # test when gradient is zero
    weights = [0.2, 0.5, 0.3]
    gradient = [0.0, 0.0, 0.0]
    learning_rate = 0.01
    assert adjust_weights(weights, gradient, learning_rate) == weights

# TODO: add more tests for edge cases and different scenarios