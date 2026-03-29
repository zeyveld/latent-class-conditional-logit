# LCL: Latent-Class Conditional Logit Estimation in Python

[![PyPI version](https://badge.fury.io/py/lcl-choice.svg)](https://badge.fury.io/py/lcl-choice)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LCL** is a high-performance Python package for estimating latent-class conditional logit models. 

Built for researchers and econometricians handling large discrete choice datasets, LCL employs **JAX** for GPU-accelerated gradient descent and Just-In-Time (JIT) compilation, alongside **Polars** for lightning-fast data management.

## 🚀 Features

* **Blazing Fast Estimation:** Core likelihood functions are written in pure JAX, allowing for seamless hardware acceleration (CPU/GPU/TPU) and automatic differentiation.
* **Modern Data Handling:** Native support for Polars DataFrames, avoiding the memory overhead and bottlenecking of traditional pandas pipelines.
* **Fail-Fast Type Checking:** Powered by `jaxtyping` and `beartype`, LCL strictly enforces tensor shapes and data types at runtime. If you add an unsupported dimension to your design matrix, LCL catches it immediately with a readable error—no more cryptic JAX compilation tracebacks!

## 📦 Installation

Although the package is imported as `lcl`, it is hosted on PyPI as `lcl-choice`.

```bash
pip install lcl-choice
```

*Note: If you plan to run LCL on a GPU, ensure you install the correct [GPU-enabled version of JAX](https://github.com/google/jax#installation) for your system.*

## 💡 Quickstart

Here is a minimal example of estimating a basic latent-class logit model using synthetic discrete choice data.

```python
import polars as pl
import jax.numpy as jnp
import lcl

# 1. Load your choice data
df = pl.DataFrame({
    "chooser_id": [1, 1, 2, 2],
    "alt_id": [1, 2, 1, 2],
    "choice": [1, 0, 0, 1],
    "price": [10.5, 12.0, 9.5, 11.0],
    "quality": [4, 5, 3, 5]
})

# 2. Format the data into JAX arrays
# (Assuming a utility function where users choose between alternatives based on price and quality)
X = jnp.array(df.select(["price", "quality"]).to_numpy())
choices = jnp.array(df["choice"].to_numpy())

# 3. Initialize and fit the model
# Estimate a model with 2 distinct latent consumer classes
model = lcl.LatentClassConditionalLogit(n_classes=2)
results = model.fit(X, choices)

print(results.summary())
```

## 🗺️ Roadmap & Future Developments

LCL is under active development. Although the core estimation engine is functional, we are actively working on expanding the package's accessibility and feature set. Upcoming milestones include:

* **Comprehensive Documentation:** We are currently building a dedicated documentation website (probably using Sphinx/MkDocs) to host detailed tutorials, mathematical appendices, and full API references.
* **Companion Paper:** A scholarly working paper detailing the econometric framework, hardware benchmarking, and Monte Carlo simulations is currently in preparation. 

**Feature Requests:** If there are specific constraints, optimization routines, or post-estimation tools you would like to see, please feel free to open a [Feature Request on our GitHub Issues page](https://github.com/your-username/latent-class-conditional-logit/issues)!

## 🛠️ Development & Contributing

We welcome contributions! LCL uses `uv` for modern, isolated dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/latent-class-conditional-logit.git
cd latent-class-conditional-logit

# Sync the virtual environment and install dev dependencies
uv sync --all-extras --dev

# Run the test suite
uv run pytest tests/
```

## 🤝 Acknowledgments

In addition to the developers behind **JAX**, **Polars**, **Beartype**, and **Jaxtyping**, we are especially grateful to the creators of the [xlogit](https://github.com/arteagac/xlogit/tree/master) package (Cristian Arteaga, JeeWoong Park, Prithvi Bhat Beeramoole, and Alexander Paz). Their highly efficient conditional logit logic profoundly influenced the architecture of this package.

## 📝 Citation

If you use LCL in your research or publications, please consider citing it:

```bibtex
@software{lcl_2026,
  author = {Jeffries, Anna and Zeyveld, Andrew},
  title = {LCL: Latent-Class Conditional Logit Estimation in Python},
  year = {2026},
  url = {https://github.com/your-username/latent-class-conditional-logit}
}
```