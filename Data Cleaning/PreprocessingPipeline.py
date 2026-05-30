from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

numeric_features = ['age', 'fare', 'pclass']
categorical_features = ['sex', 'embarked']

# Numeric preprocessor
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessor
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features)
])


def get_full_pipeline():
    """Return the full preprocessing+classifier pipeline."""
    return Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])


if __name__ == "__main__":
    # Create synthetic dataset matching expected columns so this file can run standalone.
    import numpy as np
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    n = 300
    # Numeric features
    age = rng.randint(1, 80, size=n)
    fare = rng.exponential(scale=30.0, size=n)
    pclass = rng.randint(1, 4, size=n)
    # Categorical features
    sex = rng.choice(['male', 'female'], size=n)
    embarked = rng.choice(['C', 'Q', 'S'], size=n)
    # Binary target (random for demo)
    y = rng.randint(0, 2, size=n)

    import pandas as pd

    X = pd.DataFrame({
        'age': age,
        'fare': fare,
        'pclass': pclass,
        'sex': sex,
        'embarked': embarked
    })

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = get_full_pipeline()
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f'Test accuracy: {score:.3f}')