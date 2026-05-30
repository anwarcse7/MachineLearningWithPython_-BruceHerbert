import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create synthetic train/test data
rng = np.random.RandomState(0)
# 2 features: first ~N(0,1), second ~N(10,5) with one large outlier
X_train = rng.normal(loc=[0, 10], scale=[1.0, 5.0], size=(100, 2))
X_train[0, 1] = 200.0  # add outlier in train
X_test = rng.normal(loc=[0, 10], scale=[1.0, 5.0], size=(10, 2))

# StandardScaler: fit on train, transform train/test
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

print("StandardScaler train mean (≈0):", np.round(X_train_std.mean(axis=0), 6))
print("StandardScaler train std (≈1):", np.round(X_train_std.std(axis=0), 6))

# MinMaxScaler: fit on train, transform
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
print("MinMaxScaler train min:", np.round(X_train_minmax.min(axis=0), 6))
print("MinMaxScaler train max:", np.round(X_train_minmax.max(axis=0), 6))

# RobustScaler: fit on train, transform
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
print("RobustScaler train median (≈0):", np.round(np.median(X_train_robust, axis=0), 6))

# Simple checks (will raise AssertionError if something is wrong)
assert np.allclose(X_train_std.mean(axis=0), 0, atol=1e-6)
assert np.allclose(X_train_std.std(axis=0), 1, atol=1e-6)
assert np.allclose(X_train_minmax.min(axis=0), 0, atol=1e-9)
assert np.allclose(X_train_minmax.max(axis=0), 1, atol=1e-9)

print("All scaler checks passed.")
