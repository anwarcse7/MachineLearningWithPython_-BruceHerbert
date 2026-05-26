import numpy as np
# Create a 1D array
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(array_1d)
print("1D Array Shape:")
print(array_1d.shape)
print("1D Array Data Type:")
print(array_1d.dtype)

print("----------------------------------------------")

# Create a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:")
print(array_2d)
print("2D Array Shape:")
print(array_2d.shape)
print("2D Array Data Type:")
print(array_2d.dtype)

print("----------------------------------------------")
#Useful shortcuts
zeros_array = np.zeros((3, 4)) # Creates a 3x4 array filled with zeros
print("Zeros Array:")
print(zeros_array)
random = np.random.random(10)
print("Random Array:")
print(random)
linsp = np.linspace(0, 1, 5) # Creates an array of 5 evenly spaced values between 0 and 10
print(linsp)    