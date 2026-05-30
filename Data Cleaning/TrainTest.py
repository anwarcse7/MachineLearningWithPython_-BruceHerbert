from sklearn.model_selection import train_test_split
import numpy as np


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


if __name__ == "__main__":
    # Demo data so running this file directly won't error.
    rng = np.random.RandomState(0)
    # synthetic binary classification: 2 features
    X = rng.normal(size=(200, 2))
    y = rng.randint(0, 2, size=(200,))

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f'Train size: {len(X_train)}')
    print(f'Test size:  {len(X_test)}')