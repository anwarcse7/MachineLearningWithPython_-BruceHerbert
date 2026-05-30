import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance

def get_standard_scaler():
	return StandardScaler()

def get_minmax_scaler():
	return MinMaxScaler()

def get_robust_scaler():
	return RobustScaler()

if __name__ == "__main__":
	# Quick demo so running this file directly won't error.
	rng = np.random.RandomState(0)
	X_train = rng.normal(loc=[0, 10], scale=[1.0, 5.0], size=(100, 2))
	X_train[0, 1] = 200.0  # add an outlier to illustrate RobustScaler
	X_test = rng.normal(loc=[0, 10], scale=[1.0, 5.0], size=(10, 2))

	std = get_standard_scaler()
	X_train_std = std.fit_transform(X_train)
	X_test_std = std.transform(X_test)
	print("StandardScaler train mean:", np.round(X_train_std.mean(axis=0), 6))
	print("StandardScaler train std:", np.round(X_train_std.std(axis=0), 6))

	mm = get_minmax_scaler()
	X_train_mm = mm.fit_transform(X_train)
	print("MinMaxScaler train min:", np.round(X_train_mm.min(axis=0), 6))
	print("MinMaxScaler train max:", np.round(X_train_mm.max(axis=0), 6))

	rob = get_robust_scaler()
	X_train_rob = rob.fit_transform(X_train)
	print("RobustScaler train median:", np.round(np.median(X_train_rob, axis=0), 6))

	print("Demo complete.")

# MinMaxScaler: scales to [0,1]

# RobustScaler: uses median and IQR, resistant to outliers