### 1. Vectors and Matrices

In machine learning:

- **Vector:** A 1D array representing a single sample's features or a bias term.

- **Matrix:** A 2D array representing a batch of data or a linear transformation layer (weight matrix).

Python

```
import numpy as np

# 1D Vector (d-dimensional vector)
v = np.array([2.0, 3.0, -1.0])

# 2D Matrix (m x n matrix)
W = np.array([
    [0.5, 0.2, 0.1],
    [0.9, -0.4, 0.3]
])
```

### 2. Dot Product vs. Element-Wise Multiplication

It is vital to distinguish between multiplying arrays element-by-element versus computing matrix dot products:

- **Element-Wise (`a * b`):** Hadamard product. Arrays must have matching shapes (or broadcastable shapes).

- **Dot Product (`np.dot(a, b)` or `a @ b`):** Standard linear algebra inner product.

Python

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise product
print("Element-wise:", a * b)       # [1*4, 2*5, 3*6] = [4, 10, 18]

# Vector Dot Product: sum(a_i * b_i)
dot_val = np.dot(a, b)             # 1*4 + 2*5 + 3*6 = 32
print("Dot Product:", dot_val)     # Scalar result: 32
```

### 3. Matrix Multiplication (`@` and `np.matmul`)

Matrix multiplication ($C = A \cdot B$) powers feed-forward neural network layers:

$$y = \sigma(X \cdot W + b)$$

#### Shape Rule:

To multiply matrix $A$ of shape $(m \times \mathbf{k})$ by matrix $B$ of shape $(\mathbf{k} \times n)$, inner dimensions **must match**. The resulting matrix has shape $(m \times n)$.

Python

```python
# Batch of 4 samples, each with 3 features: (4, 3)
X = np.random.randn(4, 3)

# Layer weights: 3 input features -> 2 output neurons: (3, 2)
W = np.random.randn(3, 2)

# Matrix Multiplication using the @ operator (recommended)
out_matmul = X @ W

# Equivalent using np.matmul()
out_func = np.matmul(X, W)

print("X shape:", X.shape)
print("W shape:", W.shape)
print("Output Shape (4, 2):\n", out_matmul.shape)
```

### 4. Transpose & Conjugate Swap (`.T` and `np.swapaxes`)

Transposing swaps rows and columns ($(m \times n) \rightarrow (n \times m)$). This is frequently required to align inner matrix dimensions prior to multiplication.

Python

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape: (2, 3)

A_T = A.T  # Shape: (3, 2)
print("Original Matrix (2, 3):\n", A)
print("Transposed Matrix (3, 2):\n", A_T)

# Batched Transpose for 3D Tensors (e.g., Attention Heads in Transformers)
# Swaps axis 1 and axis 2 while leaving batch dimension (axis 0) untouched
tensor_batch = np.zeros((32, 8, 64))  # (Batch, Seq_Len, Hidden_Dim)
transposed_batch = np.swapaxes(tensor_batch, 1, 2)  # (32, 64, 8)
```

### 5. Determinant and Matrix Inverse (`np.linalg.inv`, `np.linalg.det`)

The linear algebra module `np.linalg` handles specialized numerical linear algebra subroutines.

#### Matrix Inverse ($A^{-1}$):

$$A \cdot A^{-1} = I$$

A matrix is invertible **only if its determinant is non-zero** ($\det(A) \neq 0$).

Python

```python
A = np.array([
    [4.0, 7.0],
    [2.0, 6.0]
])

# 1. Determinant
det_A = np.linalg.det(A)
print("Determinant:", det_A)  # 4*6 - 7*2 = 10.0

# 2. Matrix Inverse
A_inv = np.linalg.inv(A)
print("Inverse Matrix:\n", A_inv)

# Proof: A @ A_inv equals Identity Matrix I
identity_test = np.round(A @ A_inv)
print("Verification (Identity Matrix):\n", identity_test)
```

### 6. Eigenvalues and Eigenvectors (`np.linalg.eig`)

An **eigenvector** $v$ of a matrix $A$ is a non-zero vector that does not change direction when $A$ is applied to it; it only scales by a factor called the **eigenvalue** $\lambda$:

$$A v = \lambda v$$

Eigen-decomposition forms the mathematical basis for **Principal Component Analysis (PCA)** for dimensionality reduction.

Python

```python
A = np.array([
    [4.0, 2.0],
    [1.0, 3.0]
])

# Compute eigenvalues and right eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues (λ):", eigenvalues)
print("Eigenvectors (v):\n", eigenvectors)

# Verify A @ v == λ * v for first eigenvector
v0 = eigenvectors[:, 0]
lambda0 = eigenvalues[0]

lhs = A @ v0
rhs = lambda0 * v0
print("Verification (LHS == RHS):", np.allclose(lhs, rhs))  # True
```

### 7. Vector and Matrix Norms (`np.linalg.norm`)

Norms quantify the size or magnitude of vectors and matrices. They are heavily used in ML loss regularizations ($L_1$ Lasso, $L_2$ Ridge) and measuring vector distances.



```python
v = np.array([3.0, -4.0])

# 1. L1 Norm (Manhattan Distance): sum(|x_i|)
l1_norm = np.linalg.norm(v, ord=1)
print("L1 Norm:", l1_norm)  # |3| + |-4| = 7.0

# 2. L2 Norm (Euclidean Distance / Length): sqrt(sum(x_i^2))
l2_norm = np.linalg.norm(v, ord=2)
print("L2 Norm:", l2_norm)  # sqrt(3^2 + (-4)^2) = 5.0

# 3. Matrix Frobenius Norm (used for matrix distance/regularization)
M = np.array([[1, 2], [3, 4]])
frob_norm = np.linalg.norm(M, ord='fro')
print("Frobenius Norm:", frob_norm)
```

### 8. Solving Systems of Linear Equations (`np.linalg.solve`)

Given a system of equations $A x = b$, solve for the unknown vector $x$.

Example System:

$$2x + 3y = 8$$

$$4x - y = 2$$



```python
# Coefficient matrix A and output vector b
A = np.array([[2.0, 3.0], [4.0, -1.0]])
b = np.array([8.0, 2.0])

# Direct numerical solver (faster and more numerically stable than x = A_inv @ b)
x = np.linalg.solve(A, b)

print("Solution vector [x, y]:", x)  # [1.42857143, 1.71428571]

# Verification: A @ x == b
print("Verification:", np.allclose(A @ x, b))  # True
```

> **Performance Tip:** Always use `np.linalg.solve(A, b)` instead of manually computing `np.linalg.inv(A) @ b`. `solve` uses fast LU-decomposition subroutines, avoiding the numerical errors and computational overhead of explicit matrix inversion.

### Linear Algebra Reference Cheat Sheet

| **Operation**          | **NumPy Syntax**                | **ML Application**                                       |
| ---------------------- | ------------------------------- | -------------------------------------------------------- |
| Matrix Multiply        | `A @ B` or `np.matmul(A, B)`    | Neural Network Dense Layers, Self-Attention              |
| Transpose              | `A.T` or `np.swapaxes(A, 0, 1)` | Adjusting tensor shapes for operations                   |
| Vector Length ($L_2$)  | `np.linalg.norm(v, ord=2)`      | Euclidean distance, Cosine Similarity                    |
| Regularization ($L_1$) | `np.linalg.norm(v, ord=1)`      | Feature selection / Sparsity penalties                   |
| PCA Decomposition      | `np.linalg.eig(cov_matrix)`     | Dimensionality reduction, compression                    |
| Linear System          | `np.linalg.solve(A, b)`         | Regression analytical solutions (Ordinary Least Squares) |
