import pytest
from training_utils import calculate_loss, normalize_data

# test for loss calculation function
def test_calculate_loss():
    # check loss for a simple case
    predictions = [0.2, 0.5, 0.7]
    targets = [0, 1, 1]
    expected_loss = (0.5 * (0.2 - 0)**2) + (0.5 * (0.5 - 1)**2) + (0.5 * (0.7 - 1)**2)
    
    # assert the calculated loss is close to expected
    assert abs(calculate_loss(predictions, targets) - expected_loss) < 1e-5

# test for data normalization
def test_normalize_data():
    data = [1, 2, 3, 4, 5]
    expected_normalized = [0, 0.25, 0.5, 0.75, 1]
    
    # assert normalized data is as expected
    normalized_data = normalize_data(data)
    assert len(normalized_data) == len(expected_normalized)
    
    for norm, exp in zip(normalized_data, expected_normalized):
        assert abs(norm - exp) < 1e-5

# TODO: add more tests for edge cases and different inputs