### 1. Reshaping and Transposing Data

Reshaping and transposing are fundamental for matching expected model input dimensions (e.g., swapping image channels in computer vision models from `(H, W, C)` to `(C, H, W)`).

```python
import numpy as np

# 1. Reshaping
arr = np.arange(12)               # Shape: (12,)
matrix = arr.reshape(3, 4)        # Shape: (3, 4)

# 2. Transpose (Swapping axes 0 and 1)
transposed = matrix.T             # Shape: (4, 3)
print("Original Shape:", matrix.shape)
print("Transposed Shape:", transposed.shape)

# 3. Swapaxes / Moveaxis for multi-dimensional tensors
# Example: Convert RGB Image from (Height, Width, Channels) to PyTorch format (Channels, Height, Width)
img = np.zeros((224, 224, 3))     # (H, W, C)
pytorch_img = np.transpose(img, (2, 0, 1))  # Move axis 2 to front -> (3, 224, 224)
print("PyTorch Image Shape:", pytorch_img.shape)
```

### 2. Combining Arrays: Concatenation and Stacking

| **Function**           | **Operation**                                    | **Example Use Case**                        |
| ---------------------- | ------------------------------------------------ | ------------------------------------------- |
| **`np.concatenate()`** | Joins existing arrays along an **existing axis** | Appending new training samples to a dataset |
| **`np.stack()`**       | Joins arrays along a **NEW axis**                | Combining single image tensors into a batch |
| **`np.vstack()`**      | Vertical stack (row-wise concatenation)          | Stacking rows on top of each other          |
| **`np.hstack()`**      | Horizontal stack (column-wise concatenation)     | Adding new feature columns side-by-side     |

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (rows)
concat_rows = np.concatenate((a, b), axis=0)  # Shape: (4, 2)

# Stack along a new axis (creating 3D tensor batch)
stacked = np.stack((a, b), axis=0)             # Shape: (2, 2, 2)
print("Stacked shape (Batch Size, Rows, Cols):", stacked.shape)

# vstack and hstack shortcuts
v_stacked = np.vstack((a, b))  # Same as concatenate axis=0
h_stacked = np.hstack((a, b))  # Same as concatenate axis=1
```

### 3. Splitting Arrays

`np.split` divides an array into multiple sub-arrays along an axis—commonly used to separate feature matrices from target labels, or to slice data into train/test sets.

```python
dataset = np.arange(20).reshape(10, 2)  # 10 samples, 2 features

# Split into 2 equal parts vertically (Train/Test split)
train_data, test_data = np.split(dataset, 2, axis=0)

print("Train shape:", train_data.shape)  # (5, 2)
print("Test shape:", test_data.shape)    # (5, 2)

# Split at specific row indices
p1, p2, p3 = np.array_split(dataset, [3, 7], axis=0)  # Rows 0-2, 3-6, 7-9
```

### 4. Handling Missing Values (`NaN` and `Inf`)

Real-world datasets often contain missing values (`np.nan`) or infinite values (`np.inf`). Standard NumPy operations return `nan` if even a single `nan` is present, so specialized functions are needed.

```python
data = np.array([10.0, np.nan, 30.0, np.nan, 50.0])

# 1. Detect missing values
nan_mask = np.isnan(data)
print("Is NaN:", nan_mask)  # [False, True, False, True, False]

# 2. Impute (replace) NaN values with mean using boolean indexing
clean_mean = np.nanmean(data)  # Computes mean ignoring NaNs (30.0)
data[np.isnan(data)] = clean_mean
print("Imputed Data:", data)    # [10., 30., 30., 30., 50.]

# 3. NaN-safe aggregations
data_with_nan = np.array([10, 20, np.nan])
print("Standard Sum:", np.sum(data_with_nan))     # nan
print("NaN-Safe Sum:", np.nansum(data_with_nan))  # 30.0
```

### 5. Feature Scaling: Normalization & Standardization

Models converge faster when numerical features are placed on a similar scale.

#### A. Min-Max Normalization (Rescales data to $[0, 1]$ range)

$\text{Scaled} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$



```python
X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

min_val = np.min(X)
max_val = np.max(X)

X_norm = (X - min_val) / (max_val - min_val)
print("Normalized [0, 1]:", X_norm)  # [0. , 0.25, 0.5 , 0.75, 1.  ]
```

#### B. Standardization / Z-Score (Mean $= 0$, Standard Deviation $= 1$)

$Z = \frac{X - \mu}{\sigma}$

Python

```python
mean = np.mean(X)
std = np.std(X)

X_std = (X - mean) / std
print("Standardized (Z-Score):", X_std)
print("New Mean:", np.round(np.mean(X_std)))  # 0.0
print("New Std:", np.std(X_std))               # 1.0
```

### 6. One-Hot Encoding

Machine learning models cannot process categorical labels (like class `0`, `1`, `2`) directly as continuous numbers without implying an incorrect order. One-hot encoding converts discrete class labels into binary vectors.

Python

```python
# Target class labels for 4 samples (3 total classes: 0, 1, 2)
labels = np.array([0, 2, 1, 0])
num_classes = 3

# One-hot encoding using np.eye identity matrix indexing
one_hot = np.eye(num_classes)[labels]

print("Original Labels:", labels)
print("One-Hot Encrypted Matrix:\n", one_hot)
# Output:
# [[1. 0. 0.]   <- Class 0
#  [0. 0. 1.]   <- Class 2
#  [0. 1. 0.]   <- Class 1
#  [1. 0. 0.]]  <- Class 0
```

### 7. Efficient Array Operations & In-Place Memory Execution

In large-scale AI applications, creating unnecessary temporary arrays exhausts system RAM and degrades CPU caching.

Python

```python
# 1. In-Place Operations (modifies memory directly without allocation)
a = np.ones((1000, 1000))
b = np.ones((1000, 1000))

# BAD: Allocates a new temporary matrix in memory
# c = a + b 

# GOOD: In-place addition (zero extra allocation)
a += b

# 2. Using the `out` parameter in ufuncs
result = np.empty_like(a)
np.add(a, b, out=result)  # Writes directly into pre-allocated buffer

# 3. Vectorization over Loops (Always prefer ufuncs over Python loops)
# Avoid looping over rows; pass the whole matrix to NumPy functions directly.
```
