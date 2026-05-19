# mamimo — Changes & Migration Guide

## What Changed and Why

The original mamimo package (v0.4.3) was written against scikit-learn 1.0–1.1,
numpy 1.22, and pandas 1.4. All three have had breaking API changes since then.
This update makes the package compatible with current versions and with
**Databricks Runtime 17.3** (Python 3.12).

---

## Package Version Requirements

| Package | Old requirement | New requirement | Why |
|---|---|---|---|
| Python | `^3.9` | `^3.12` | DBR 17.3 ships Python 3.12 |
| scikit-learn | `^1.0.2` | `^1.6.0` | `validate_data()` added in 1.6 |
| numpy | `^1.22.3` | `^1.24.0` | API stability; 2.x supported |
| pandas | `^1.4.2` | `^2.0.0` | Copy-on-write semantics in 2.x |
| scipy | _(unlocked)_ | `^1.10.0` | Explicit pin for reproducibility |

---

## Source File Changes

### `mamimo/linear_model.py`

**Problem:** `check_X_y`, `check_array`, and the instance method
`self._check_n_features()` were removed from `sklearn.BaseEstimator` in
sklearn 1.6. Calling them raises `AttributeError` or `ImportError`.

**Fix:**
- Removed imports: `check_X_y`, `check_array`
- Added import: `validate_data` from `sklearn.utils.validation`
- `check_X_y(X, y)` + `self._check_n_features(X, reset=True)` →
  `validate_data(self, X, y)` (sets `n_features_in_` automatically)
- `check_array(X)` + `self._check_n_features(X, reset=False)` in `predict()` →
  `validate_data(self, X, reset=False)`

---

### `mamimo/saturation.py`

**Problem:** `self._validate_data()` (instance method) removed in sklearn 1.6.
`check_array` + `self._check_n_features()` likewise removed.

**Fix:**
- Removed import: `check_array`
- Added import: `validate_data`
- `self._validate_data(X, dtype=FLOAT_DTYPES)` → `validate_data(self, X, dtype=FLOAT_DTYPES)`
- `check_array(X)` + `self._check_n_features(X, reset=False)` →
  `validate_data(self, X, reset=False, dtype=FLOAT_DTYPES)`

---

### `mamimo/carryover.py`

**Problem:** Same as `saturation.py`, plus a stale `from sklearn.utils import check_array`.

**Fix:**
- Removed: `from sklearn.utils import check_array`
- Added import: `validate_data`
- Same `_validate_data` / `check_array` / `_check_n_features` replacements as above

---

### `mamimo/time_utils.py`

**Problem:** `PowerTrend` class had the same `_validate_data`, `check_array`,
and `_check_n_features` issues.

**Fix:** Same pattern applied to `PowerTrend.fit()` and `PowerTrend.transform()`.

---

### `pyproject.toml`

Updated all dependency version pins to match the above requirements.
Added `scipy` as an explicit dependency (it was an implicit transitive
dependency before).

---

### `demo.py` (new file)

A [marimo](https://marimo.io) demo notebook added to the repo root. It
demonstrates the full MMM workflow end-to-end:
- Dataset loading and preview
- Saturation curve comparison (BoxCox, Hill, Exponential)
- Carryover / adstock effect visualisation
- Model fitting with `ColumnTransformer` + `LinearRegression(positive=True)`
- Channel contribution breakdown via `mamimo.analysis.breakdown`
- Validation summary (package versions, R², contribution integrity check)

Run it locally with:
```bash
pip install marimo
marimo edit demo.py
```

---

## Note on `OneHotEncoder(sparse=False)` in README Examples

The existing README examples use `OneHotEncoder(sparse=False, ...)`. In
sklearn 1.2+ the `sparse` parameter was renamed to `sparse_output`. Update
any usage to `sparse_output=False` to avoid a `FutureWarning` (it becomes
an error in sklearn 1.6+).

---

## Databricks Runtime 17.3 Integration

### Pre-installed packages on DBR 17.3

DBR 17.3 ships Python 3.12 and includes scikit-learn, numpy, pandas, and
scipy pre-installed. Check the installed sklearn version before running:

```python
import sklearn
print(sklearn.__version__)   # must be >= 1.6.0 for this mamimo version
```

If the installed version is below 1.6.0, upgrade it at the top of your
notebook before any other imports:

```python
%pip install --upgrade "scikit-learn>=1.6.0"
dbutils.library.restartPython()
```

---

### Option 1 — Databricks Repos (recommended)

Add this repo to Databricks via **Repos → Add Repo**. The repo root is
automatically placed on `sys.path`, so all imports work without any path
manipulation:

```python
# In any Databricks notebook in the same repo — no sys.path needed
from mamimo.datasets import load_fake_mmm
from mamimo.carryover import ExponentialCarryover
from mamimo.saturation import ExponentialSaturation, HillSaturation
from mamimo.linear_model import LinearRegression
from mamimo.time_utils import add_date_indicators, PowerTrend
from mamimo.analysis import breakdown
```

The repo structure must have `mamimo/` (the Python package directory) at the
root level — which it does in this layout:

```
repo-root/
  mamimo/          ← Python package (importable)
    __init__.py
    analysis.py
    carryover.py
    linear_model.py
    saturation.py
    time_utils.py
    datasets/
  demo.py
  requirements.txt
  pyproject.toml
```

---

### Option 2 — Install as a wheel onto a cluster

Build and upload the package, then install it on your cluster or in a notebook:

```bash
# Local: build the wheel
cd /path/to/mamimo
pip install build
python -m build --wheel
# Produces: dist/mamimo-0.4.3-py3-none-any.whl
```

Upload the `.whl` to a Databricks Volume or DBFS, then in your notebook:

```python
%pip install /Volumes/catalog/schema/volume/mamimo-0.4.3-py3-none-any.whl
dbutils.library.restartPython()

# Imports are then identical to local usage
from mamimo.datasets import load_fake_mmm
```

Or install directly onto the cluster via **Compute → Libraries → Install New
→ PyPI / DBFS** so all notebooks on that cluster have access without a
per-notebook `%pip install`.

---

### Option 3 — DBFS path injection (quick and dirty)

If you just want to copy the `mamimo/` package folder to DBFS without
installing it as a proper package:

```python
import sys
sys.path.insert(0, '/dbfs/FileStore/mamimo')   # adjust path as needed

from mamimo.datasets import load_fake_mmm
# ... rest of imports unchanged
```

This is the least robust option — prefer Repos or a wheel for shared clusters.

---

### Converting `demo.py` for Databricks

`demo.py` is a marimo notebook and will not run directly in Databricks.
Convert it to a standard Databricks/Jupyter notebook by copying each cell's
code block into a Databricks `%python` cell. Replace `mo.md(...)` cells with
`displayHTML(...)` or just use Markdown cells:

```python
# marimo cell → Databricks equivalent
# mo.md("## Title")  →  (use a Markdown cell in Databricks)

# mo.ui.table(df)  →
display(df)

# Everything else (numpy, pandas, sklearn, mamimo calls) is identical
```

---

### Cluster init script (optional)

To guarantee sklearn >= 1.6 is available on every cluster restart without
a per-notebook `%pip` call, add a cluster init script:

```bash
#!/bin/bash
/databricks/python/bin/pip install --upgrade "scikit-learn>=1.6.0"
```
