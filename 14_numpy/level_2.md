### 1. Mathematical Operations (Vectorized Math)

In pure Python, adding two lists requires looping through every element. In NumPy, arithmetic operations are **vectorized**—they execute in parallel at C-speed across entire arrays element-by-element.



```python
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

# Element-wise operations
print(a + b)        # [11 22 33 44]
print(a - b)        # [9 18 27 36]
print(a * b)        # [10 40 90 160]  (Hadamard Product / Element-wise multiplication)
print(a / b)        # [10. 10. 10. 10.]
print(a ** 2)       # [100 400 900 1600]

# Essential Universal Functions (ufuncs) for AI
x = np.array([0, 1, 2, 3])
print(np.exp(x))    # Exponential e^x (used in Softmax & Sigmoid activations)
print(np.log(x + 1))# Natural log (used in Loss Functions like Cross-Entropy)
print(np.sqrt(x))   # Square root (used in Normalization / Standard Deviation)
```

### 2. The Axis Concept ⭐ (Crucial for AI Data Structures)

Understanding **`axis`** is the single most critical step in transitioning from basic programming to tensor manipulation in PyTorch/TensorFlow.

- **`axis=0`:** Runs **downwards** across rows (collapses vertical elements).

- **`axis=1`:** Runs **across** columns (collapses horizontal elements).

```python
            axis 1 (across columns)
               --->
        [ [ 1,  2,  3 ],
axis 0  [ 4,  5,  6 ] ]
 (down)
```



```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# axis=0: Collapse rows (sum each column)
print(matrix.sum(axis=0))  # [1+4, 2+5, 3+6] = [5, 7, 9]

# axis=1: Collapse columns (sum each row)
print(matrix.sum(axis=1))  # [1+2+3, 4+5+6] = [6, 15]
```

> **Rule of Thumb:** Specifying `axis=k` eliminates dimension `k` from the array shape!
> 
> If `shape` is `(2, 3)` and you compute `.sum(axis=0)`, the resulting shape is `(3,)`.

### 3. Aggregations (`sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`)

Aggregations reduce arrays to summary statistics—used everywhere in calculating loss, accuracy, and normalizations.



```python
scores = np.array([
    [80, 95, 70],
    [60, 85, 90]
])

print("Total Sum:", np.sum(scores))            # 475
print("Mean per student (axis=1):", np.mean(scores, axis=1)) # [81.66, 78.33]
print("Max per test (axis=0):", np.max(scores, axis=0))      # [80, 95, 90]
print("Standard Deviation:", np.std(scores))

# argmax and argmin return the INDEX of extreme values (used for Classification outputs)
predictions = np.array([0.1, 0.75, 0.15])     # Output probabilities for 3 classes
predicted_class = np.argmax(predictions)      # Returns index 1 (Class 1 has highest prob)
print("Predicted Class Index:", predicted_class)
```

### 4. Broadcasting ⭐ (NumPy's Core Superpower)

**Broadcasting** allows NumPy to perform operations on arrays of **different shapes** without allocating unnecessary memory copies.

#### The 2 Rules of Broadcasting:

Two dimensions are compatible when:

1. They are **equal**, OR

2. One of them is **1**.

```python
Example: Normalizing a Dataset (Subtract Mean from Matrix)

Matrix (3, 2):             Mean Vector (1, 2):
[[ 10, 200 ],              [[ 20, 200 ]]  <-- Broadcasts downwards to shape (3, 2)
 [ 20, 200 ],       -
 [ 30, 200 ]]
```



```python
# Case 1: Scalar Broadcasting (1D vector with a single number)
arr = np.array([1, 2, 3])
print(arr + 10)  # [11, 12, 13] -> 10 is broadcast across all elements

# Case 2: 2D Matrix with 1D Vector
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape (2, 3)

row_vec = np.array([10, 20, 30])  # Shape (3,) -> matches last dimension

# NumPy automatically stretches row_vec into 2 rows to match (2, 3)
print(matrix + row_vec)
# Output:
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 5. Comparisons and Boolean Logic

Comparisons produce boolean arrays that allow element-by-element evaluation.



```python
data = np.array([12, 45, 67, 89, 23])

# Boolean operators: & (AND), | (OR), ~ (NOT)
# Important: Always wrap individual conditions in parentheses!
condition = (data > 20) & (data < 70)
print(condition)            # [False  True  True False  True]
print(data[condition])      # [45 67 23]

# Checking overall conditions
print(np.any(data > 80))    # True (at least one element satisfies)
print(np.all(data > 80))    # False (not all elements satisfy)
```

### 6. Conditional Selection with `np.where()`

`np.where(condition, x, y)` acts like a vectorized `if-else` statement:

$$\text{Output} = \begin{cases} x & \text{if condition is True} \\ y & \text{if condition is False} \end{cases}$$

Python

```python
scores = np.array([45, 78, 92, 30, 65])

# If score >= 50 output 'Pass', else 'Fail'
results = np.where(scores >= 50, 'Pass', 'Fail')
print(results)  # ['Fail' 'Pass' 'Pass' 'Fail' 'Pass']

# Common ML pattern: Relu Activation function implementation
# Force all negative values to become 0
raw_inputs = np.array([-2.5, 3.1, -0.4, 5.0])
relu_output = np.where(raw_inputs > 0, raw_inputs, 0)
print(relu_output)  # [0.  3.1 0.  5. ]
```

### 7. Sorting Arrays



```python
arr = np.array([40, 10, 30, 20])

# 1. np.sort() returns a sorted COPY
sorted_arr = np.sort(arr)
print(sorted_arr)  # [10, 20, 30, 40]

# 2. np.argsort() returns the INDICES that would sort the array (Crucial for Top-K predictions)
indices = np.argsort(arr)
print("Sorted Indices:", indices)  # [1 3 2 0] -> arr[1]=10 is smallest, arr[0]=40 is largest
```

### 8. Searching Arrays (`np.nonzero`, `np.searchsorted`)



```python
arr = np.array([10, 0, 25, 0, 50])

# Find indices of non-zero elements
nz_indices = np.nonzero(arr)
print(nz_indices)  # (array([0, 2, 4]),)

# Searchsorted: Find insertion points in a SORTED array to maintain order
sorted_data = np.array([10, 20, 30, 40, 50])
insert_pos = np.searchsorted(sorted_data, 25)
print("Insert 25 at index:", insert_pos)  # Index 2
```

### 9. Random Number Generation (`np.random`)

Used extensively for parameter initialization, data shuffling, and dropout layers.



```python
# Set seed for reproducibility (makes random generation deterministic across runs)
np.random.seed(42)

# Generate uniform random floats in range [0.0, 1.0)
rand_float = np.random.rand(2, 3)

# Generate values from Standard Normal Distribution (mean=0, std=1)
rand_norm = np.random.randn(2, 3)

# Generate random integers
rand_ints = np.random.randint(low=1, high=10, size=(3, 3))

# Shuffling data (In-place shuffling along axis 0)
features = np.array([[1, 2], [3, 4], [5, 6]])
np.random.shuffle(features)
print("Shuffled Batch:\n", features)
```
