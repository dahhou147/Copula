# Copula

A Python library for modeling and simulating multivariate dependencies using copulas, specifically Gaussian and Student-t copulas. This library provides tools for dependency modeling, correlation analysis, and risk assessment in financial and statistical applications.

## Features

- **Gaussian Copula**: Fit and sample from a Gaussian copula using empirical covariance.
- **Student Copula**: Fit and sample from a Student-t copula using Kendall's tau and maximum likelihood estimation for degrees of freedom.
- **Utility Functions**: Matrix manipulation, normalization, correlation conversion, and semi-definite positive matrix operations.
- **Statistical Tools**: Kendall's tau to linear correlation conversion, covariance approximation, and maximum likelihood estimation.

## Use Cases

### 1. Financial Risk Modeling
- **Portfolio Risk Assessment**: Model dependencies between different financial assets
- **Credit Risk Modeling**: Analyze correlations between default probabilities
- **Market Risk Analysis**: Simulate joint movements of market variables

### 2. Statistical Analysis
- **Dependency Modeling**: Capture complex multivariate dependencies beyond linear correlations
- **Data Generation**: Generate synthetic data with specified dependency structures
- **Correlation Analysis**: Convert between different correlation measures (Kendall's tau to linear correlation)

### 3. Matrix Operations
- **Semi-definite Positive Matrix Conversion**: Ensure valid covariance matrices
- **Matrix Normalization**: Standardize correlation matrices
- **Symmetric Matrix Construction**: Build correlation matrices from parameter arrays

### 4. Monte Carlo Simulation
- **Scenario Generation**: Create realistic scenarios for stress testing
- **Risk Quantification**: Estimate tail dependencies and extreme events
- **Model Validation**: Test model assumptions and parameter stability

## Installation

Install the required dependencies (add these to your `requirements.txt`):

```
numpy
scipy
```

Then install with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Copula Usage

```python
from copulas import GaussianCopula, StudentCopula
import numpy as np

# Example data
samples = np.random.rand(1000, 3)

# Gaussian Copula
gauss = GaussianCopula().fit(samples)
gauss_samples = gauss.rvs(100)

# Student Copula
student = StudentCopula().fit(samples)
student_samples = student.rvs(100)
```

### Utility Functions

```python
from copulas import array_to_symm_matrix, kendall_to_linear_correlation, matrix_to_sdp

# Convert correlation array to symmetric matrix
corr_array = [0.1, 0.2, 0.13, 0.43, 0.9, 0.81]  # 3x3 matrix parameters
corr_matrix = array_to_symm_matrix(corr_array)

# Convert Kendall's tau to linear correlation
tau_correlation = kendall_to_linear_correlation(x, y)

# Ensure matrix is semi-definite positive
sdp_matrix = matrix_to_sdp(corr_matrix)
```

### Financial Application Example

```python
import numpy as np
from copulas import StudentCopula, array_to_symm_matrix, matrix_to_sdp

# Define correlation structure for 3 assets
corr_params = [0.3, 0.5, 0.2]  # Correlation between asset pairs
cov_matrix = array_to_symm_matrix(corr_params)
cov_matrix = matrix_to_sdp(cov_matrix)

# Generate historical data (simulated)
historical_data = np.random.rand(1000, 3)

# Fit Student-t copula
copula = StudentCopula().fit(historical_data)

# Generate scenarios for stress testing
scenarios = copula.rvs(10000)

# Analyze tail dependencies
tail_events = scenarios[scenarios > 0.95]
```

### Risk Management Example

```python
from copulas import GaussianCopula
import numpy as np

# Portfolio returns data (3 assets)
returns_data = np.random.randn(1000, 3)

# Fit Gaussian copula
copula = GaussianCopula().fit(returns_data)

# Generate Monte Carlo scenarios
n_scenarios = 10000
scenarios = copula.rvs(n_scenarios)

# Calculate portfolio risk metrics
portfolio_returns = np.mean(scenarios, axis=1)
var_95 = np.percentile(portfolio_returns, 5)
print(f"95% VaR: {var_95:.4f}")
```

## File Overview

- `copulas.py`: Main implementation containing:
  - `GaussianCopula`: Gaussian copula implementation
  - `StudentCopula`: Student-t copula implementation
  - Utility functions for matrix operations and correlation conversions
  - Parameter classes for copula specifications
- `requirements.txt`: List of dependencies (numpy, scipy)
- `__init__.py`: Marks the directory as a Python package
- `.gitignore`: Git ignore configuration

## Key Functions

### Copula Classes
- `GaussianCopula.fit()`: Fits using empirical covariance
- `StudentCopula.fit()`: Fits using Kendall's tau and MLE for degrees of freedom
- `Copula.rvs()`: Generates random variates from fitted copula

### Utility Functions
- `array_to_symm_matrix()`: Converts parameter array to symmetric matrix
- `kendall_to_linear_correlation()`: Converts Kendall's tau to linear correlation
- `matrix_to_sdp()`: Ensures matrix is semi-definite positive
- `Normalizing()`: Normalizes correlation matrix
- `approxi_var_cov()`: Approximates variance-covariance matrix
- `approxi_kendall_cov()`: Approximates Kendall's covariance matrix

## Development

- The code is written in Python and uses `numpy` and `scipy` for numerical computations and statistical functions
- To run the example in `copulas.py`, execute:
  ```bash
  python copulas.py
  ```

## Mathematical Background

This library implements:
- **Copula Theory**: Separates marginal distributions from dependency structure
- **Gaussian Copula**: Uses multivariate normal distribution for dependency modeling
- **Student-t Copula**: Captures tail dependencies using multivariate t-distribution
- **Kendall's Tau**: Rank correlation measure for robust dependency estimation
- **Maximum Likelihood Estimation**: Optimizes degrees of freedom for Student-t copula

## License

Specify your license here. 