import pytest
from training_utils import calculate_loss, update_model_weights

# testing loss calculation
def test_calculate_loss():
    predictions = [0.2, 0.5, 0.8]
    targets = [0, 1, 1]
    expected_loss = 0.5  # this is a simple mock value, adjust if needed
    loss = calculate_loss(predictions, targets)
    
    # just checking that the loss is close to what we expect
    assert abs(loss - expected_loss) < 0.01

# testing model weights update
def test_update_model_weights():
    weights = [0.1, 0.2, 0.3]
    gradients = [0.01, 0.02, 0.01]
    learning_rate = 0.1
    expected_weights = [0.099, 0.198, 0.299]  # simple calculation here
    
    updated_weights = update_model_weights(weights, gradients, learning_rate)
    
    # check if the weights are updated correctly
    for w, ew in zip(updated_weights, expected_weights):
        assert abs(w - ew) < 0.01

# TODO: add more tests for edge cases and invalid inputs