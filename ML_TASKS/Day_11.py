# ==========================================
# TASK 1 : TENSOR OPERATIONS
# Difference Between Tensor and NumPy Arrays
# Performance Comparison
# ==========================================

# Import Libraries
import numpy as np
import tensorflow as tf
import time

# ==========================================
# NumPy Array Example
# ==========================================

print("===================================")
print("NUMPY ARRAY")
print("===================================")

numpy_array = np.array([[1, 2],
                        [3, 4]])

print("NumPy Array:")
print(numpy_array)

print("\nType:")
print(type(numpy_array))

# ==========================================
# Tensor Example
# ==========================================

print("\n===================================")
print("TENSOR")
print("===================================")

tensor = tf.constant([[1, 2],
                      [3, 4]])

print("Tensor:")
print(tensor)

print("\nType:")
print(type(tensor))

# ==========================================
# Addition Operation
# ==========================================

print("\n===================================")
print("ADDITION OPERATION")
print("===================================")

a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])

print("NumPy Addition:")
print(a_np + b_np)

a_tf = tf.constant([1, 2, 3])
b_tf = tf.constant([4, 5, 6])

print("\nTensor Addition:")
print(tf.add(a_tf, b_tf))

# ==========================================
# Multiplication Operation
# ==========================================

print("\n===================================")
print("MULTIPLICATION OPERATION")
print("===================================")

print("NumPy Multiplication:")
print(a_np * b_np)

print("\nTensor Multiplication:")
print(tf.multiply(a_tf, b_tf))

# ==========================================
# Performance Comparison
# ==========================================

print("\n===================================")
print("PERFORMANCE COMPARISON")
print("===================================")

size = 2000

array1 = np.random.rand(size, size)
array2 = np.random.rand(size, size)

# NumPy Performance
start = time.time()

result_numpy = np.dot(array1, array2)

end = time.time()

numpy_time = end - start

print("NumPy Matrix Multiplication Time:")
print(numpy_time, "seconds")

# TensorFlow Performance
tensor1 = tf.constant(array1, dtype=tf.float32)
tensor2 = tf.constant(array2, dtype=tf.float32)

start = time.time()

result_tensor = tf.matmul(tensor1, tensor2)

end = time.time()

tensor_time = end - start

print("\nTensorFlow Matrix Multiplication Time:")
print(tensor_time, "seconds")

# ==========================================
# Automatic Differentiation
# ==========================================

print("\n===================================")
print("AUTOMATIC DIFFERENTIATION")
print("===================================")

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)

print("x =", x.numpy())
print("y =", y.numpy())
print("dy/dx =", gradient.numpy())

# ==========================================
# Conclusion
# ==========================================

print("\n===================================")
print("CONCLUSION")
print("===================================")

print("1. NumPy arrays are used for numerical computations.")
print("2. Tensors are used in Machine Learning and Deep Learning.")
print("3. Tensors support GPU acceleration.")
print("4. Tensors support automatic differentiation.")
print("5. Tensors are more suitable for training neural networks.")
