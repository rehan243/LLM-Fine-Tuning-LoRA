import pytest
from training_utils import prepare_data, calculate_accuracy

def test_prepare_data():
    # test with mock input
    raw_data = ["text 1", "text 2", "text 3"]
    expected_output = ["text 1", "text 2", "text 3"]  # assuming no changes
    processed_data = prepare_data(raw_data)
    assert processed_data == expected_output

def test_calculate_accuracy():
    # test with sample predictions and labels
    predictions = [1, 0, 1, 1]
    labels = [1, 0, 0, 1]
    expected_accuracy = 0.5  # 2 correct out of 4
    accuracy = calculate_accuracy(predictions, labels)
    assert accuracy == expected_accuracy

def test_prepare_data_empty():
    # test with empty data
    raw_data = []
    expected_output = []
    processed_data = prepare_data(raw_data)
    assert processed_data == expected_output

# TODO: add more tests for edge cases and different inputs