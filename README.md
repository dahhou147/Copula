# Copula

A Python library for modeling and simulating multivariate dependencies using copulas, specifically Gaussian and Student-t copulas.

## Features

- **Gaussian Copula**: Fit and sample from a Gaussian copula using empirical covariance.
- **Student Copula**: Fit and sample from a Student-t copula using Kendall’s tau and maximum likelihood estimation for degrees of freedom.
- Utility functions for matrix manipulation, normalization, and correlation conversion.

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

```python
from copulas import GaussianCopula, StudentCopula

# Example data
import numpy as np
samples = np.random.rand(1000, 3)

# Gaussian Copula
gauss = GaussianCopula().fit(samples)
gauss_samples = gauss.rvs(100)

# Student Copula
student = StudentCopula().fit(samples)
student_samples = student.rvs(100)
```

## File Overview

- `copulas.py`: Main implementation of copula models and utility functions.
- `requirements.txt`: List of dependencies (should include numpy, scipy).
- `__init__.py`: Marks the directory as a Python package.
- `.gitignore`: (Currently empty) Add files/folders to ignore in git.

## Development

- The code is written in Python and uses `numpy` and `scipy` for numerical computations and statistical functions.
- To run the example in `copulas.py`, execute:
  ```bash
  python copulas.py
  ```

## License

Specify your license here. 