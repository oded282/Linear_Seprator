import loader
import linear_separators
import test
from sklearn.model_selection import train_test_split
import sys

FIRST_ARG = 1
SEC_ARG = 2


def main():
    try:
        train_x, train_y = loader.load_data(sys.argv[FIRST_ARG], sys.argv[SEC_ARG])
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)
        weights = linear_separators.run_algorithm(train_x, train_y, linear_separators.passive_aggressive)
    except IOError as e1:
        print(e1)
        sys.exit(1)

    except Exception as e2:
        print("Error: algorithm failed to run\n", e2)
        sys.exit(1)

    test.run_test(weights, test_x, test_y)


if __name__ == "__main__":
    main()
