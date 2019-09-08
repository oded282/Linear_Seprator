import numpy as np

ETA = 0.01
LAMDA = 0.01
ROW = 3
COlUMN = 8


def calculate_tao(true_label, predicted_label, weights, example):
    return (max(0, 1 - np.dot(weights[true_label], example) + np.dot(weights[predicted_label], example))) / (
            2 * (np.power(np.linalg.norm(example), 2)))


def run_algorithm(train_x, train_y, update_rule):
    weights = np.zeros((ROW, COlUMN), dtype='f8')
    for iteration in range(10):
        for example, true_label in zip(train_x, train_y):
            predicted_label = int(np.argmax(weights.dot(example)))
            if predicted_label != true_label:
                update_rule(predicted_label, true_label, weights, example)
    return weights


def perceptron(predicted_label, true_label, weights, example):
    weights[true_label] += ETA * example
    weights[predicted_label] -= ETA * example


def passive_aggressive(predicted_label, true_label, weights, example):
    if np.count_nonzero(example) == 0:
        raise ZeroDivisionError('Error: Invalid division by zero occur')

    tao = calculate_tao(predicted_label, true_label, weights, example)
    weights[true_label] += tao * example
    weights[predicted_label] -= tao * example


def svm(predicted_label, true_label, weights, example):
    weights[true_label] = (1 - LAMDA * ETA) * weights[true_label] + ETA * example
    weights[predicted_label] = (1 - LAMDA * ETA) * weights[predicted_label] - ETA * example
