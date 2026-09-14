### 1. What is NumPy and Why Machine Learning Uses It

**NumPy** (Numerical Python) is the core library for scientific computing in Python.

In pure Python, if you multiply two lists element-by-element using a `for` loop, Python performs **dynamic type checking** and pointer lookups on every single step. This makes p ure Python slow for large numerical datasets.

NumPy solves this by introducing contiguous memory structures in C:

- **Vectorization:** Operations run in parallel compiled C loops without explicit Python `for` loops.

- **Contiguous Memory:** Array elements sit next to each other in memory, maximizing CPU cache performance.

Python

```
import time
import numpy as np

# Pure Python vs NumPy benchmark
size = 10_000_000
py_list1, py_list2 = list(range(size)), list(range(size))
np_arr1, np_arr2 = np.arange(size), np.arange(size)

# Python list addition
start = time.time()
py_result = [x + y for x, y in zip(py_list1, py_list2)]
print(f"Python list time: {time.time() - start:.4f} seconds")

# NumPy vectorized addition
start = time.time()
np_result = np_arr1 + np_arr2
print(f"NumPy vector time: {time.time() - start:.4f} seconds")
```

*NumPy is typically 20× to 100× faster than pure Python loops.*

### 2. Installing & Importing NumPy

Bash

```
pip install numpy
```

Python

```
import numpy as np

# Verify version
print(np.__version__)
```

### 3. NumPy Arrays (`ndarray`) & How to Create Them

The core building block of NumPy is the **`ndarray`** (N-dimensional array).

Python

```
import numpy as np

# From Python lists
a_1d = np.array([1, 2, 3])                # 1D Array (Vector)
a_2d = np.array([[1, 2, 3], [4, 5, 6]])    # 2D Array (Matrix)

# Built-in creation functions
zeros = np.zeros((3, 3))                   # 3x3 array filled with 0.0
ones = np.ones((2, 4))                     # 2x4 array filled with 1.0
full = np.full((2, 2), 7)                  # 2x2 array filled with 7
identity = np.eye(3)                       # 3x3 Identity matrix (1s on diagonal)

# Sequences and Ranges
seq = np.arange(0, 10, 2)                  # [0, 2, 4, 6, 8] (start, stop, step)
linspace = np.linspace(0, 1, 5)            # 5 evenly spaced numbers from 0 to 1

# Random generation (Essential for initializing neural network weights)
rand_uniform = np.random.rand(2, 3)        # Uniform distribution [0, 1)
rand_normal = np.random.randn(2, 3)        # Standard Normal distribution (mean 0, std 1)
```

### 4. Essential Array Attributes (`shape`, `ndim`, `size`, `dtype`)

These four attributes give you the exact metadata of any tensor/array:

| **Attribute** | **What it tells you**             | **ML Context**                                         |
| ------------- | --------------------------------- | ------------------------------------------------------ |
| `ndim`        | Number of dimensions (axes)       | 1D = Vector, 2D = Matrix, 3D = Image, 4D = Image Batch |
| `shape`       | Tuple showing length of each axis | Example: `(32, 3, 224, 224)` = batch of 32 RGB images  |
| `size`        | Total number of elements          | Product of all values in `shape`                       |
| `dtype`       | Data type of elements             | `float32`, `int64`, etc.                               |

Python

```
arr = np.array([[[1, 2, 3], [4, 5, 6]]])

print("Dimensions (ndim):", arr.ndim)   # 3
print("Shape:", arr.shape)             # (1, 2, 3)
print("Total elements (size):", arr.size)# 6
print("Data Type (dtype):", arr.dtype)  # int64 or int32 depending on OS
```

### 5. Indexing and Slicing

Slicing syntax follows standard `[start:stop:step]` across each dimension, separated by commas: `array[row_slice, col_slice]`.

Python

```
matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 95, 98, 99]
])

# 1. Select single element (row index 1, column index 2)
print(matrix[1, 2])        # 70

# 2. Select an entire row or column
print(matrix[0, :])        # First row: [10, 20, 30, 40]
print(matrix[:, 1])        # Second column: [20, 60, 95]

# 3. Sub-matrix extraction (Rows 0 to 1, Cols 1 to 2)
print(matrix[0:2, 1:3])
# Output:
# [[20, 30]
#  [60, 70]]

# 4. Boolean Indexing (Filtering) - Crucial for data preprocessing
mask = matrix > 50
print(matrix[mask])        # [60, 70, 80, 90, 95, 98, 99]
```

### 6. Changing Array Shape (`reshape`, `ravel`, `flatten`)

Reshaping is constant work in AI (e.g., flattening an image matrix into a 1D feature vector before passing it into a fully connected layer).

Python

```
x = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape into a 3x4 matrix
grid = x.reshape(3, 4)

# Automatic dimension inferencing using -1
# NumPy automatically calculates that 12 / 2 = 6
auto_grid = x.reshape(2, -1)  # Shape (2, 6)

# Flatten back to 1D
flat_view = grid.ravel()      # Returns a view (fast)
flat_copy = grid.flatten()    # Returns a deep copy
```

> **Note:** The total `size` (number of elements) must remain unchanged when reshaping. You cannot reshape 12 elements into `(3, 5)`.

### 7. Memory Mechanics: Copy vs View

This is where bugs frequently hide in numerical pipelines.

- **View:** Shares the exact same underlying memory block. Modifying a view **mutates the original array**.

- **Copy:** Allocates brand-new memory. Modifying a copy **does not affect the original array**.

Python

```
base_arr = np.array([1, 2, 3, 4, 5])

# Slicing creates a VIEW by default in NumPy
slice_view = base_arr[1:4]
slice_view[0] = 999

print(base_arr)  # Output: [  1, 999,   3,   4,   5] -> Original changed!

# Explicit COPY allocation
base_arr2 = np.array([1, 2, 3, 4, 5])
explicit_copy = base_arr2[1:4].copy()
explicit_copy[0] = 999

print(base_arr2) # Output: [1, 2, 3, 4, 5] -> Original unchanged!
```

### 8. Data Types (`dtype`) and Precision in AI

In machine learning:

- **`float32` (Single Precision):** Standard for deep learning model training (balances memory and numerical accuracy).

- **`float16` / `bfloat16` (Half Precision):** Used in mixed-precision training on modern GPUs to save VRAM.

- **`int64` / `int32`:** Used for discrete labels, indices, and token IDs.

Python

```
# Specifying dtype at creation
arr_f32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)

# Type Casting using .astype()
arr_int = arr_f32.astype(np.int32)  # [1, 2, 3] (truncates decimals)

print("Data type:", arr_f32.dtype)
print("Bytes per element:", arr_f32.itemsize) # 4 bytes for float32
```
