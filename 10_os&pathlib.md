# Python File Paths: `os` vs `pathlib`

## Phase 1: The Core Problem (How Computers Store Files)

Your computer stores everything in a tree of folders (directories).

Imagine you have a file named `cow.jpg`. It lives inside a folder called `images`, which lives inside `dataset`:

```
dataset/
└── images/
    └── cow.jpg
```

To find `cow.jpg`, your operating system needs an address. That address is called a **Path**:

```
dataset/images/cow.jpg
```

### The Big Problem for Python Programmers

- On **Linux/macOS**, folder paths use forward slashes: `dataset/images/cow.jpg`
- On **Windows**, folder paths use backslashes: `dataset\images\cow.jpg`

If you write code on a Mac using `/`, and then run that code on a Windows computer (or vice versa), your program can crash or silently fail because the path format doesn't match.

---

## Phase 2: The Two Ways Python Solves This

To prevent path errors, Python gives us two tools:

1. **The Old Way** (`import os`) — treats paths as plain text strings.
2. **The Modern Way** (`from pathlib import Path`) — treats paths as smart objects.

---

## Phase 3: The Old Way (`os`)

In older Python code, you'll see the `os` module. It treats every folder location as simple text.

### 1. Checking if a folder exists and creating it

If you try to save a file into a folder that doesn't exist, Python crashes. You have to check first.

```python
import os

folder_name = "my_dataset"

# Ask Python: "Does 'my_dataset' exist on the hard drive?"
if not os.path.exists(folder_name):
    # If it doesn't exist, create it!
    os.makedirs(folder_name)
```

### 2. Joining folder names together safely

Instead of writing `"dataset" + "/" + "images"`, use `os.path.join()`. This function automatically figures out if your computer is Windows or Linux and adds the correct slash (`/` or `\`).

```python
import os

folder = "my_dataset"
filename = "cow.jpg"

# os.path.join builds the correct path automatically
full_path = os.path.join(folder, filename)

print(full_path)
# On Mac/Linux, prints: my_dataset/cow.jpg
# On Windows, prints:   my_dataset\cow.jpg
```

### 3. Other common `os`/`os.path` functions

```python
import os

os.path.isfile("my_dataset/cow.jpg")   # True if it's a file
os.path.isdir("my_dataset")            # True if it's a folder
os.path.getsize("my_dataset/cow.jpg")  # File size in bytes
os.remove("my_dataset/cow.jpg")        # Delete a file
os.rename("old.jpg", "new.jpg")        # Rename a file
os.getcwd()                            # Get current working directory
```

---

## Phase 4: The Modern Way (`pathlib`)

Starting in Python 3.4, Python introduced `pathlib`. Instead of forcing you to treat paths as dumb text strings, `pathlib` gives you a `Path` object that understands files natively.

### 1. Creating a Path object

```python
from pathlib import Path

# Turn a simple string into a smart Path object
my_path = Path("my_dataset/images/cow.jpg")
```

### 2. Why Path objects are amazing (the "parts" of a file)

Once a path is a `Path` object, you can extract any part of the file name without doing string manipulation:

```python
from pathlib import Path

file = Path("my_dataset/images/cow.jpg")

print(file.name)    # "cow.jpg"                  (full filename)
print(file.stem)    # "cow"                       (name WITHOUT extension)
print(file.suffix)  # ".jpg"                      (extension only)
print(file.parent)  # "my_dataset/images"          (containing folder)
print(file.parts)   # ('my_dataset', 'images', 'cow.jpg')  (path as a tuple)
```

### 3. Joining paths with the `/` operator

With `pathlib`, you don't need `os.path.join()`. You can literally use the `/` division operator to join folders:

```python
from pathlib import Path

folder = Path("my_dataset")
subfolder = folder / "images"
file_path = subfolder / "cow.jpg"

print(file_path)
# my_dataset/images/cow.jpg (or my_dataset\images\cow.jpg on Windows)
```

### 4. Creating folders easily

```python
from pathlib import Path

folder = Path("my_dataset/train_data")

# parents=True  -> if "my_dataset" doesn't exist, create it too
# exist_ok=True -> if "train_data" already exists, don't throw an error
folder.mkdir(parents=True, exist_ok=True)
```

### 5. Checking things about a path

```python
from pathlib import Path

file = Path("my_dataset/images/cow.jpg")

file.exists()      # True if the path exists (file OR folder)
file.is_file()      # True if it's a file
file.is_dir()        # True if it's a folder
file.resolve()        # Converts to a full absolute path
```

### 6. Renaming and modifying paths without string hacks

```python
from pathlib import Path

file = Path("my_dataset/images/cow.jpg")

# Swap the extension
print(file.with_suffix(".png"))   # my_dataset/images/cow.png

# Swap the filename entirely (keeps the same folder)
print(file.with_name("buffalo.jpg"))  # my_dataset/images/buffalo.jpg
```

### 7. Listing everything inside a folder

```python
from pathlib import Path

folder = Path("my_dataset")

for item in folder.iterdir():
    print(item, "-> folder" if item.is_dir() else "-> file")
```

### 8. Useful shortcuts

```python
from pathlib import Path

Path.cwd()   # Current working directory, as a Path object
Path.home()  # The user's home directory, as a Path object
```

---

## Phase 5: Searching for Files (`glob` and `rglob`)

As an AI engineer, you'll often have a folder containing 10,000 images, and you only want to grab the `.jpg` files.

`pathlib` gives us two tools for this: `.glob()` and `.rglob()`.

### 1. `.glob()` — search ONLY the current folder

`*` is a wildcard meaning "match anything." `*.jpg` means "match anything that ends with `.jpg`."

```python
from pathlib import Path

folder = Path("my_dataset")

# Find all JPG files inside "my_dataset" (top level only)
for image_file in folder.glob("*.jpg"):
    print(image_file.name)
```

### 2. `.rglob()` — search the current folder AND all subfolders (recursive)

If images are hidden inside nested subfolders (`my_dataset/cats/img1.jpg`, `my_dataset/dogs/img2.jpg`), `.glob()` won't find them. Use `.rglob()` (Recursive Glob) instead:

```python
from pathlib import Path

folder = Path("my_dataset")

# Look inside "my_dataset" AND every subfolder inside it
for image_file in folder.rglob("*.jpg"):
    print(image_file)
    # my_dataset/cats/img1.jpg
    # my_dataset/dogs/img2.jpg
```

### 3. Matching multiple extensions

`glob`/`rglob` only take one pattern at a time, so a common trick is to loop over extensions:

```python
from pathlib import Path

folder = Path("my_dataset")
extensions = ["*.jpg", "*.png", "*.jpeg"]

image_files = []
for ext in extensions:
    image_files.extend(folder.rglob(ext))

print(len(image_files), "images found")
```

---

## Phase 6: `os` vs `pathlib` — Side-by-Side Cheat Sheet

| Task                 | `os` way                                     | `pathlib` way                                |
| -------------------- | -------------------------------------------- | -------------------------------------------- |
| Join paths           | `os.path.join(a, b)`                         | `a / b`                                      |
| Check if exists      | `os.path.exists(p)`                          | `Path(p).exists()`                           |
| Check if file        | `os.path.isfile(p)`                          | `Path(p).is_file()`                          |
| Check if folder      | `os.path.isdir(p)`                           | `Path(p).is_dir()`                           |
| Create folder(s)     | `os.makedirs(p, exist_ok=True)`              | `Path(p).mkdir(parents=True, exist_ok=True)` |
| Get filename         | `os.path.basename(p)`                        | `Path(p).name`                               |
| Get folder name      | `os.path.dirname(p)`                         | `Path(p).parent`                             |
| Get extension        | `os.path.splitext(p)[1]`                     | `Path(p).suffix`                             |
| List files in folder | `os.listdir(p)`                              | `Path(p).iterdir()`                          |
| Search by pattern    | `glob.glob(p + "/*.jpg")`                    | `Path(p).glob("*.jpg")`                      |
| Recursive search     | `glob.glob(p + "/**/*.jpg", recursive=True)` | `Path(p).rglob("*.jpg")`                     |

**Rule of thumb:** for any new project, prefer `pathlib`. Use `os` mainly when reading legacy code or working with libraries that specifically expect string paths (in which case, wrap with `str(my_path)`).

---

## Common Mistakes to Avoid

- **Hardcoding slashes**: writing `"dataset" + "/" + "images"` instead of `os.path.join()` or the `/` operator. Breaks on Windows.
- **Forgetting `exist_ok=True`**: calling `.mkdir()` on a folder that already exists will raise `FileExistsError` without it.
- **Confusing relative vs. absolute paths**: `Path("cow.jpg")` is relative to wherever the script is *run from*, not where the script *file* lives. Use `Path(__file__).resolve().parent` to anchor paths to the script's own location.
- **Using `.glob()` when you meant `.rglob()`**: `.glob()` won't look inside subfolders — a very common source of "why isn't my script finding any images" bugs.

---

## Summary Checklist for Your Notes

- **Path**: the address of a file on a computer's drive.
- **`os` module**: the older way — works with plain text strings.
- **`pathlib` module**: the modern, recommended way — turns path strings into smart objects.

**Key `pathlib` properties:**

- `.name` → filename with extension (`"photo.png"`)
- `.stem` → filename without extension (`"photo"`)
- `.suffix` → extension only (`".png"`)
- `.parent` → containing directory
- `.parts` → path broken into a tuple of pieces

**Key `pathlib` methods:**

- `folder / "file.txt"` → joins paths safely across Windows/Linux
- `path.mkdir(parents=True, exist_ok=True)` → creates directories safely
- `path.exists()`, `.is_file()`, `.is_dir()` → check what a path points to
- `path.with_suffix(".png")` / `.with_name("new.jpg")` → modify a path without string hacks
- `folder.iterdir()` → list everything directly inside a folder
- `folder.glob("*.jpg")` → find all `.jpg` files in a folder (top level only)
- `folder.rglob("*.jpg")` → find all `.jpg` files in a folder **and all subfolders**
