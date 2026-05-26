import time
import numpy as np

size = 1000000
a = np.random.randn(size)
b = np.random.randn(size)

# Python loop
start = time.time()
result = [a[i] + b[i] for i in range(size)]
print(f'Python loop time:, {time.time() - start:.3f}s')

#Numpy vectorization
start = time.time()
result = a + b
print(f'Numpy vectorization time:, {time.time() - start:.4f}s')