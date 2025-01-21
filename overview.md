Organizing a project for implementing stochastic models like Black-Scholes, Heston, Dupire, and SABR requires a clear and modular structure to ensure maintainability, scalability, and ease of use. Your idea of separating concerns into classes for calibration, graphics, and models is a good starting point. Below, I’ll provide a detailed structure and recommendations for organizing your project.

---

### **Project Structure**
Here’s a suggested structure for your project:

```
option_pricing_project/
│
├── models/                     # Folder for model implementations
│   ├── black_scholes.py        # Black-Scholes model
│   ├── heston.py               # Heston model
│   ├── dupire.py               # Dupire model
│   ├── sabr.py                 # SABR model
│   └── __init__.py             # Make the folder a Python package
│
├── calibration/                # Folder for calibration logic
│   ├── calibration.py          # Calibration class
│   └── __init__.py             # Make the folder a Python package
│
├── visualization/              # Folder for visualization logic
│   ├── graphics.py             # Graphics class
│   └── __init__.py             # Make the folder a Python package
│
├── data/                       # Folder for market data
│   ├── market_data.csv         # Example market data file
│   └── __init__.py             # Make the folder a Python package
│
├── utils/                      # Folder for utility functions
│   ├── helpers.py              # Helper functions (e.g., for simulations, calculations)
│   └── __init__.py             # Make the folder a Python package
│
├── tests/                      # Folder for unit tests
│   ├── test_black_scholes.py   # Tests for Black-Scholes
│   ├── test_heston.py          # Tests for Heston
│   └── __init__.py             # Make the folder a Python package
│
├── app.py                      # Streamlit app for user interaction
├── main.py                     # Main script to run simulations/calibrations
└── requirements.txt            # List of dependencies
```

---

### **Key Components**

#### 1. **Models**
Each model (Black-Scholes, Heston, Dupire, SABR) should be implemented in its own Python file under the `models/` folder. Each file should contain a class representing the model, with methods for:
- Simulating asset price paths.
- Calculating option prices.
- Any other model-specific functionality.

Example for `black_scholes.py`:
```python
import numpy as np

class BlackScholes:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def simulate_paths(self, n_simulations, n_steps):
        # Simulate asset price paths
        dt = self.T / n_steps
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.S0
        for t in range(1, n_steps + 1):
            z = np.random.normal(size=n_simulations)
            paths[:, t] = paths[:, t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        return paths

    def price_option(self, option_type='call'):
        # Calculate option price
        pass
```

---

#### 2. **Calibration**
The `calibration/` folder should contain a `calibration.py` file with a `Calibration` class. This class will handle:
- Loading market data.
- Calibrating model parameters to market data (e.g., using optimization techniques like `scipy.optimize`).
- Storing calibrated parameters.

Example:
```python
from scipy.optimize import minimize

class Calibration:
    def __init__(self, market_data):
        self.market_data = market_data

    def calibrate_black_scholes(self):
        # Calibrate Black-Scholes parameters
        pass

    def calibrate_heston(self):
        # Calibrate Heston parameters
        pass
```

---

#### 3. **Visualization**
The `visualization/` folder should contain a `graphics.py` file with a `Graphics` class. This class will handle:
- Plotting simulated price paths.
- Visualizing calibration results.
- Comparing model outputs.

Example:
```python
import matplotlib.pyplot as plt

class Graphics:
    def plot_paths(self, paths):
        plt.figure(figsize=(10, 6))
        for path in paths:
            plt.plot(path)
        plt.title("Simulated Asset Price Paths")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.show()
```

---

#### 4. **Data**
The `data/` folder should contain market data files (e.g., CSV files) and utility functions for loading and preprocessing data.

---

#### 5. **Utils**
The `utils/` folder should contain helper functions used across the project, such as:
- Random number generation.
- Mathematical calculations.
- File I/O operations.

Example:
```python
import numpy as np

def generate_gaussian_random_numbers(n):
    return np.random.normal(size=n)
```

---

#### 6. **Tests**
The `tests/` folder should contain unit tests for each model and utility function. Use a testing framework like `pytest`.

Example for `test_black_scholes.py`:
```python
from models.black_scholes import BlackScholes

def test_black_scholes_simulation():
    model = BlackScholes(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    paths = model.simulate_paths(n_simulations=1000, n_steps=252)
    assert paths.shape == (1000, 253)
```

---

#### 7. **Streamlit App**
The `app.py` file will serve as the entry point for the Streamlit app. It will allow users to:
- Select a model.
- Input parameters.
- Run simulations or calibrations.
- Visualize results.

Example:
```python
import streamlit as st
from models.black_scholes import BlackScholes

st.title("Option Pricing Models")

model_name = st.selectbox("Select Model", ["Black-Scholes", "Heston", "Dupire", "SABR"])
S0 = st.number_input("Initial Price (S0)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T)", value=1.0)

if st.button("Run Simulation"):
    if model_name == "Black-Scholes":
        model = BlackScholes(S0=S0, K=K, T=T, r=0.05, sigma=0.2)
        paths = model.simulate_paths(n_simulations=1000, n_steps=252)
        st.line_chart(paths.mean(axis=0))
```

---

### **Recommendations**
1. **Modularity**:
   - Keep each model in a separate file for clarity and maintainability.
   - Use classes to encapsulate functionality (e.g., `BlackScholes`, `Heston`, etc.).

2. **Reusability**:
   - Use utility functions for common tasks (e.g., random number generation, file I/O).

3. **Testing**:
   - Write unit tests for each model and utility function to ensure correctness.

4. **Documentation**:
   - Add docstrings to classes and methods.
   - Include a `README.md` file explaining how to use the project.

5. **Streamlit Integration**:
   - Use Streamlit for interactive visualization and user input.

---

### **Final Thoughts**
This structure is scalable and modular, making it easy to add new models or functionality in the future. By separating concerns into different folders and files, you ensure that your project remains organized and easy to navigate.