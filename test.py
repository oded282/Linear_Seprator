import numpy as np


def run_test(weights, test_x, test_y):
    """
        Summery:
            This function test's the weights function that we trained earlier and
            measures the amount of loss.
        Arguments:
            test_x [str]: The train_x array 328x8.
            test_y [str]: The train_y array 328X1.
        Returns:
            None

    """
    loss = 0
    for example, true_label in zip(test_x, test_y):
        predicted_label = int(np.argmax(weights.dot(example)))
        if predicted_label != true_label:
            loss += 1
    print(loss / len(test_y))
