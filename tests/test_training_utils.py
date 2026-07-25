import pytest
from src.training_utils import calculate_loss, update_weights

def test_calculate_loss():
    # testing with some dummy values
    predictions = [0.5, 0.7, 0.2]
    targets = [0, 1, 0]
    
    loss = calculate_loss(predictions, targets)
    expected_loss = 0.5 # replace with actual expected value
    assert abs(loss - expected_loss) < 0.01, f"expected loss: {expected_loss}, got: {loss}"

def test_update_weights():
    # testing weight update
    weights = [0.1, 0.2, 0.3]
    gradients = [0.01, 0.02, 0.01]
    learning_rate = 0.1
    
    updated_weights = update_weights(weights, gradients, learning_rate)
    expected_weights = [0.099, 0.198, 0.299] # replace with actual expected values
    for w, ew in zip(updated_weights, expected_weights):
        assert abs(w - ew) < 0.001, f"expected weight: {ew}, got: {w}"

# TODO: add more tests for edge cases and other functions in training_utils