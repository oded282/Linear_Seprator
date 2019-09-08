import numpy as np


def run_test(weights, test_x, test_y):
    loss = 0
    for example, true_label in zip(test_x, test_y):
        predicted_label = int(np.argmax(weights.dot(example)))
        if predicted_label != true_label:
            loss += 1
    print(loss / len(test_y))
