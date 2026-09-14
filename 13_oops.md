### Importance for AI/ML Engineers

- **Importance Score:** **10/10**

- **Where It Is Used in AI/ML:**
  
  - **Deep Learning Frameworks:** Every single neural network architecture in PyTorch (`torch.nn.Module`) or TensorFlow is written as a Python class.
  
  - **Custom Dataset Pipelines:** When loading custom images or text for training, you subclass PyTorch's `Dataset` class.
  
  - **MLOps & Pipeline Wrappers:** Encapsulating model training, evaluation, and saving logic into clean, reusable application classes.


# Object-Oriented Programming for AI/ML Engineers

## Part 1: The Core Theory of OOP

In procedural programming, you write isolated functions that pass data around. In **Object-Oriented Programming**, you bundle data (state) and functions (behaviors) together inside a single container called a **Class**.

- **Class**: The architectural blueprint (e.g., "Car Blueprint").
- **Object (Instance)**: The physical realization built from that blueprint (e.g., "My red Tesla").
- **Constructor (`__init__`)**: A special initialization method that runs automatically the exact second an object is created.
- **`self`**: A reference variable pointing directly to the current specific object you are working with.
- **Methods**: Functions defined inside a class that perform actions using the object's data.
- **Inheritance**: A mechanism where a new class takes on all attributes and methods of an existing parent class, allowing you to extend or modify functionality without rewriting code.

---

## Part 2: Deep-Dive Code Example & Line-by-Line Breakdown

Let's examine a custom class representing an AI model manager.

```python
class Model:
    # 1. The Constructor (__init__)
    def __init__(self, name: str, learning_rate: float):
        # Instance variables (unique to every individual object created)
        self.name = name
        self.learning_rate = learning_rate
        self.is_trained = False

    # 2. A Class Method (Behavior)
    def train(self, epochs: int):
        print(f"Starting training for {self.name} with LR: {self.learning_rate}...")
        # Modifying the object's internal state
        self.is_trained = True
        print(f"Training complete! Status: {self.is_trained}")

    # 3. Another Method
    def predict(self, image_tensor):
        if not self.is_trained:
            raise RuntimeError(f"Cannot run inference. Model {self.name} is not trained yet!")

        print(f"Running inference using {self.name}...")
        return "Prediction Output: [Class: Cow, Confidence: 0.94]"

# --- Instantiation (Creating Objects from the Class) ---

# Create Object 1
model_a = Model(name="ResNet50", learning_rate=0.001)

# Create Object 2
model_b = Model(name="YOLOv8", learning_rate=0.0005)

# Call methods on Object 1
model_a.train(epochs=10)
result = model_a.predict(image_tensor="dummy_image_data")
print(result)
```

### Line-by-Line Explanation

- `class Model:` — Declares a new blueprint named `Model`. By convention, class names use `CapitalizedWords` (PascalCase).
- `def __init__(self, name, learning_rate):` — The constructor. Python calls this automatically when you write `Model(...)`.
- `self.name = name` — Binds the incoming string argument (`"ResNet50"`) to `self.name`. This creates an **instance variable** that stays attached to that specific object throughout its lifecycle.
- `self.is_trained = False` — Sets a default starting state for every new model object.
- `model_a = Model(...)` — Instantiates a concrete object in memory. Behind the scenes, Python passes `model_a` into `self` automatically.

---

## Part 3: Instance Variables vs. Class Variables

A distinction that trips people up early on: variables can belong to a specific *object*, or to the *class itself* (shared across every object).

```python
class Model:
    # Class variable — shared by EVERY instance of Model
    framework = "PyTorch"

    def __init__(self, name):
        # Instance variable — unique to THIS object
        self.name = name

model_a = Model("ResNet50")
model_b = Model("YOLOv8")

print(model_a.framework)  # "PyTorch"
print(model_b.framework)  # "PyTorch"  (same value, shared)

Model.framework = "TensorFlow"  # change it on the class...
print(model_a.framework)         # "TensorFlow" (both instances see the change)
```

**Rule of thumb:** if a value should differ per object (a model's name, its weights), it's an instance variable defined with `self.` inside `__init__`. If it's a constant shared by all objects of that type (a default config, a version tag), it's a class variable.

---

## Part 4: `@staticmethod` and `@classmethod`

Not every method needs access to `self`. Two decorators handle the exceptions:

```python
class Model:
    model_count = 0

    def __init__(self, name):
        self.name = name
        Model.model_count += 1
    
    # Doesn't need self OR the class — pure utility function, just lives here for organization
    @staticmethod
    def is_valid_learning_rate(lr: float) -> bool:
        return 0 < lr < 1
    
    # Needs the CLASS, not a specific instance — often used for alternate constructors
    @classmethod
    def from_config(cls, config: dict):
        return cls(name=config["name"])

print(Model.is_valid_learning_rate(0.0)
