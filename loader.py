import numpy as np
import os

MALE = 0.01
FEMALE = 0.02
INFANT = 0.03
SEX_INDEX = 0


def shuffle_unison(train_x, train_y):
    if len(train_x) != len(train_y):
        raise Exception('Error: train_x and train_y size does not match.')
    permutation = np.random.permutation(train_x.shape[0])
    train_x = train_x[permutation]
    train_y = train_y[permutation]
    return train_x, train_y


def set_gender_value(data):
    for abalone in data:
        if abalone[SEX_INDEX] == 'M':
            abalone[SEX_INDEX] = MALE
        elif abalone[SEX_INDEX] == 'F':
            abalone[SEX_INDEX] = FEMALE
        elif abalone[SEX_INDEX] == 'I':
            abalone[SEX_INDEX] = INFANT
        else:
            raise Exception("Invalid gender")
    return data.astype(np.float)


def load_data(train_x, train_y):
    if not os.path.isfile(train_x) or not os.path.isfile(train_y):
        raise IOError('Error: One or both of the files does not exist.')
    data = np.genfromtxt(train_x, delimiter=',', dtype='unicode')
    train_x = set_gender_value(data)
    train_y = np.loadtxt(train_y, dtype='i4')

    return shuffle_unison(train_x, train_y)
