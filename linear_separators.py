import numpy as np

ETA = 0.01
LAMDA = 0.01
ROW = 3
COlUMN = 8


def calculate_tao(true_label, predicted_label, weights, example) -> int:
    """
        Summery:
            The function calculate the Tao value as given in the Passive_Aggressive algorithm.
        Arguments:
            predicted_label [int]: The age that the weight function predict for the specific example
            true_label [int]: The true age prediction for the specific example.
            weights [np.array]: The weight function np array 8x3.
            example [np.array]: One of the example, np.array 8x1
        Returns:
            The Tao value.

    """
    return (max(0, 1 - np.dot(weights[true_label], example) + np.dot(weights[predicted_label], example))) / (
            2 * (np.power(np.linalg.norm(example), 2)))


def run_algorithm(train_x, train_y, update_rule) -> np.array:
    """
        Summery:
            The function iterate all over the example and test the weight function,
            if the prediction was wrong we update the weight function by the update_rule
            that the user piked.
        Arguments:
            train_x [str]: The train_x array 2957X8.
            train_y [str]: The train_y array 2957x1.
            update_rule [func]: This func responsible to the updating rule, the user can use one of the three
                                algorithm implemented for the update rule: Perceptron, SVM or Passive_Aggressive.
        Returns:
            weights [np.array]: The weight function np array 8x3.

    """
    weights = np.zeros((ROW, COlUMN), dtype='f8')
    for iteration in range(10):
        for example, true_label in zip(train_x, train_y):
            predicted_label = int(np.argmax(weights.dot(example)))
            if predicted_label != true_label:
                update_rule(predicted_label, true_label, weights, example)
    return weights


def perceptron(predicted_label, true_label, weights, example):
    """
        Summery:
            Change the weight function according to the Perceptron update rule.
        Arguments:
            predicted_label [int]: The age that the weight function predict for the specific example
            true_label [int]: The true age prediction for the specific example.
            weights [np.array]: The weight function np array 8x3.
            example [np.array]: One of the example, np.array 8x1
        Returns:
            None.

    """
    weights[true_label] += ETA * example
    weights[predicted_label] -= ETA * example


def passive_aggressive(predicted_label, true_label, weights, example):
    """
        Summery:
            Change the weight function according to the Pa update rule.
        Arguments:
            predicted_label [int]: The age that the weight function predict for the specific example
            true_label [int]: The true age prediction for the specific example.
            weights [np.array]: The weight function np array 8x3.
            example [np.array]: One of the example, np.array 8x1
        Returns:
            None.

    """
    if np.count_nonzero(example) == 0:
        raise ZeroDivisionError('Error: Invalid division by zero occur')

    tao = calculate_tao(predicted_label, true_label, weights, example)
    weights[true_label] += tao * example
    weights[predicted_label] -= tao * example


def svm(predicted_label, true_label, weights, example):
    """
        Summery:
            Change the weight function according to the svm update rule.
        Arguments:
            predicted_label [int]: The age that the weight function predict for the specific example
            true_label [int]: The true age prediction for the specific example.
            weights [np.array]: The weight function np array 8x3.
            example [np.array]: One of the example, np.array 8x1
        Returns:
            None.

    """
    weights[true_label] = (1 - LAMDA * ETA) * weights[true_label] + ETA * example
    weights[predicted_label] = (1 - LAMDA * ETA) * weights[predicted_label] - ETA * example
