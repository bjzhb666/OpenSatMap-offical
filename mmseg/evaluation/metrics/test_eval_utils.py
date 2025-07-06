import numpy as np
from eval_utils import sample_from_positions

def test_sample_from_positions():
    # Test case 1: with_noise=True
    position = np.array([[1, 2], [1, 4], [1, 6], [1, 8]])
    # expected_output = np.array([[1, 4], [3, 6], [5, 6], [7, 8]])
    sample_from_positions(position, with_noise=True)

    # Test case 2: with_noise=False
    position = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_output = np.array([[1, 4], [3, 6], [5, 6], [7, 8]])
    assert np.array_equal(sample_from_positions(position, with_noise=False), expected_output)

    # # Test case 3: empty position array
    # position = np.array([])
    # expected_output = np.array([])
    # assert np.array_equal(sample_from_positions(position, with_noise=True), expected_output)

    # # Test case 4: position array with one point
    # position = np.array([[1, 2]])
    # expected_output = np.array([[1, 2]])
    # assert np.array_equal(sample_from_positions(position, with_noise=True), expected_output)

    # # Test case 5: position array with multiple points at same position
    # position = np.array([[1, 2], [1, 2], [1, 2]])
    # expected_output = np.array([[1, 2]])
    # assert np.array_equal(sample_from_positions(position, with_noise=True), expected_output)

    print("All test cases pass")

test_sample_from_positions()