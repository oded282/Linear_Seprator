import numpy as np
import os

MALE = 0.01
FEMALE = 0.02
INFANT = 0.03
GENDER_INDEX = 0


def shuffle_unison(train_x, train_y):
    """
        Summery:
            This function shuffle the arrays correspondent.
        Arguments:
            train_x [str]: The train_x array 3286X8.
            train_y [str]: The train_y array 3286X1.
        Returns:
            shuffled_train_x [np.array]: Shuffled train_x array 3286X8 type=float.
            shuffled_train_x [np.array]: Shuffled train_x array 3286X1 type=int.
    """
    if len(train_x) != len(train_y):
        raise Exception('Error: train_x and train_y size does not match.')
    permutation = np.random.permutation(train_x.shape[0])
    train_x = train_x[permutation]
    train_y = train_y[permutation]
    return train_x, train_y


def set_gender_value(data) -> np.array:
    """
        Summery:
            This function set the gender feature to be float type.
        Arguments:
            data [np.array]: The train_x data array 3286X8.
        Returns:
            data [np.array]: The train_x data array with gender's set values 3286X8.
    """
    for abalone in data:
        if abalone[GENDER_INDEX] == 'M':
            abalone[GENDER_INDEX] = MALE
        elif abalone[GENDER_INDEX] == 'F':
            abalone[GENDER_INDEX] = FEMALE
        elif abalone[GENDER_INDEX] == 'I':
            abalone[GENDER_INDEX] = INFANT
        else:
            raise Exception("Invalid gender")
    return data.astype(np.float)


def load_data(train_x, train_y) -> (np.array, np.array):
    """
        Summery:
            This function check if the files really exist and load the data from the file to arrays.
        Arguments:
            train_x [str]: The train_x file name.
            train_x [str]: The train_y file name.
        Returns:
            shuffled_train_x [np.array]: Shuffled train_x array 3286X8 type=float.
            shuffled_train_x [np.array]: Shuffled train_x array 3286X1 type=int.
    """

    if not os.path.isfile(train_x) or not os.path.isfile(train_y):
        raise IOError('Error: One or both of the files does not exist.')
    data = np.genfromtxt(train_x, delimiter=',', dtype='unicode')
    train_x = set_gender_value(data)
    train_y = np.loadtxt(train_y, dtype='i4')

    return shuffle_unison(train_x, train_y)
