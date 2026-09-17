In Pandas, **`pd.Series`** is a 1D labeled array and **`pd.DataFrame`** is a 2D table composed of multiple Series as columns. Under the hood, memory management and performance depend heavily on the array backend driving these structures.

### Memory Representations: NumPy vs. PyArrow

Historically, Pandas stored numerical data as 1D **NumPy `ndarray`** blocks in a structure called the `BlockManager`. While fast for pure numerical linear algebra, NumPy backends have two major drawbacks in machine learning workflows:

1. **No native missing values for integers/booleans:** NumPy relies on `NaN` (a float) for missing values. Storing a single `NaN` in an integer column forces the entire column to cast to `float64`, consuming double the memory and breaking integer pipelines.

2. **String overhead:** Strings are stored as Python object pointers (`object` dtype), incurring massive memory overhead and frequent cache misses.

Starting with Pandas 2.0+, **PyArrow** backend integration introduced native zero-copy operations, Apache Arrow memory layouts, and strict type safety.

| **Feature**              | **NumPy Backend (engine='c')**                | **PyArrow Backend (engine='pyarrow')**                    |
| ------------------------ | --------------------------------------------- | --------------------------------------------------------- |
| **Missing Values**       | Casts integers to `float64` for `NaN`         | Native Bitmasking (keeps `int64` intact)                  |
| **String Storage**       | Python `object` pointers (high memory)        | Arrow `StringArray` (contiguous bytes)                    |
| **Zero-Copy Conversion** | Copies memory when passing to PyTorch/XGBoost | Zero-copy memory sharing via Arrow buffers                |
| **Speed**                | Standard                                      | Significantly faster for string parsing & multi-threading |

Python

```
import pandas as pd
import numpy as np

# Traditional NumPy backend (Notice integer column converting to float due to NaN)
s_numpy = pd.Series([1, 2, None], dtype="Int64") # Int64 nullable or float64

# PyArrow backend (Native PyArrow types)
s_arrow = pd.Series([1, 2, None], dtype="int64[pyarrow]")
print(s_arrow.dtype)  # int64[pyarrow]
```

### High-Performance Ingestion

When preparing large datasets for ML training, raw disk I/O and memory exhaustion are common bottlenecks.

#### 1. Fast File Formats: CSV vs. Parquet

`Parquet` is a columnar storage format optimized for compressed, binary data. It reduces file size drastically and allows reading specific columns without loading the full file into RAM.

Python

```
# Loading CSV using PyArrow engine for multi-threaded fast parsing
df_csv = pd.read_csv("dataset.csv", engine="pyarrow")

# Loading Parquet (Preferred for ML - faster read times and preserves precise schema/dtypes)
df_parquet = pd.read_parquet("dataset.parquet", columns=["feature_1", "feature_2", "target"])
```

#### 2. Chunking Large CSV Files

When a dataset exceeds available RAM, read it in chunks to perform incremental cleaning or batch processing:

Python

```
# Process a 10GB CSV in 100,000-row chunks
chunk_size = 100_000
for chunk in pd.read_csv("huge_dataset.csv", chunksize=chunk_size):
    # Perform mini-batch cleaning or filter rows
    filtered_chunk = chunk[chunk["age"] > 18]
    # Pass directly to incremental training model (e.g., SGDClassifier.partial_fit)
```

### Dataset Inspection for ML

Before building feature pipelines, you must audit dataset shape, memory footprint, data types, and missingness.

Python

```
# 1. Dimensions
print(df.shape)  # Returns (rows, columns)

# 2. Schema and Memory Usage
df.info(memory_usage="deep")  # Shows detailed RAM consumption including object strings

# 3. Quick Data Type Check
print(df.dtypes)

# 4. Numerical Distribution Summary
print(df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))

# 5. Missing Value Counts
missing_counts = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({"count": missing_counts, "percent": missing_percentage})
```

### Selection & Indexing

Pandas provides distinct methods for selecting subsets of features or samples.

#### 1. Selection with `.loc[]` (Label-Based)

Used to select rows and columns by their labels or boolean masks.

Python

```
# Selection Syntax: df.loc[row_labels, column_labels]
subset = df.loc[0:100, ["age", "income", "target"]]

# Conditional Filtering via .loc
high_earners = df.loc[df["income"] > 75000, ["user_id", "target"]]
```

#### 2. Selection with `.iloc[]` (Position-Based)

Used to select rows and columns by integer positions (`0` to `N-1`), identical to standard Python array indexing.

Python

```
# Selection Syntax: df.iloc[row_positions, column_positions]
X = df.iloc[:, :-1]  # Extract all rows, all columns except the last (Features)
y = df.iloc[:, -1]   # Extract all rows, only the last column (Target)

# First 5 rows, first 3 columns
sample = df.iloc[0:5, 0:3]
```

#### 3. Boolean Indexing & Vectorized Filtering

Combine multiple condition masks using bitwise operators: `&` (AND), `|` (OR), `~` (NOT). *Always wrap individual conditions in parentheses.*

Python

```
# Filter rows where age is over 25 AND income is greater than 50k
mask = (df["age"] > 25) & (df["income"] > 50000)
filtered_df = df[mask]

# Using .isin() for categorical feature filtering
val_categories = ["A", "B", "C"]
filtered_cats = df[df["category"].isin(val_categories)]
```
