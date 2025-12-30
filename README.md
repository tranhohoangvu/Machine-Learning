# Intro to Machine Learning – Final Project

> **Coursework repository (2024)** — includes report, notebooks, and datasets for 3 tasks:
> 1) Optimization methods (GD/Momentum/Adagrad + extensions)  
> 2) Stock price forecasting (sequence length = 60) with neural networks & baselines  
> 3) MNIST image classification using CNN

---

## Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Datasets](#datasets)
- [Tasks & Methodology](#tasks--methodology)
  - [Q1 — Optimization Methods](#q1--optimization-methods)
  - [Q2 — Stock Open Price Forecasting](#q2--stock-open-price-forecasting)
  - [Q3 — MNIST CNN Classification](#q3--mnist-cnn-classification)
- [How to Run](#how-to-run)
- [Reproducibility Notes](#reproducibility-notes)
- [Authors](#authors)
- [License](#license)

---

## Project Overview

This repo stores the deliverables of a Machine Learning final project:
- A **PDF report**
- Three **Jupyter notebooks** (one for each question)
- Two **CSV datasets**

Each notebook is designed to be run independently.

---

## Repository Structure

Recommended structure after reorganizing:

```
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ report/
│  └─ 52200207_52200214_52200216.pdf
├─ notebooks/
│  ├─ cau1_optimization_methods.ipynb
│  ├─ cau2_stock_forecasting.ipynb
│  └─ cau3_mnist_cnn.ipynb
└─ data/
   ├─ HousingData.csv
   └─ data_src_2.csv
```

> If you keep the original file names, everything still works — the structure above is only for cleanliness when publishing on GitHub.

---

## Tech Stack

- **Python** (recommended 3.10+)
- **NumPy / Pandas** for data processing
- **Matplotlib** for visualization
- **Scikit-learn** for metrics & baseline models
- **TensorFlow / Keras** for neural networks
- **Jupyter Notebook / JupyterLab** for experiments

---

## Datasets

### 1) `HousingData.csv` (506 rows × 14 columns)
Columns:
- Features: `CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT`
- Target: `MEDV`

Used in **Q1** for testing optimization algorithms on a regression problem.

### 2) `data_src_2.csv` (6816 rows × 10 columns)
Columns:
- `Date, Adj Close, Close, High, Low, Open, Volume, Industry, Ticker, GDP`

Used in **Q2** for time-series forecasting of **Open** prices by ticker.

---

## Tasks & Methodology

### Q1 — Optimization Methods

Notebook: `notebooks/cau1_optimization_methods.ipynb`

Goal: implement and compare optimization methods for minimizing a regression loss.

Implemented methods include (as seen in the notebook):
- **Batch Gradient Descent**
- **Stochastic Gradient Descent (SGD)**
- **Mini-batch Gradient Descent**
- **Momentum**
- **Adagrad**
- *(Extra extensions for comparison)* **RMSProp**, **Adam**

Workflow (high-level):
1. Load `HousingData.csv`
2. Standardize/prepare features and target
3. Train a simple regression model (linear regression style)
4. Track **loss per epoch** for each optimizer
5. Plot loss curves and compare final loss values

Outputs:
- Loss curves (matplotlib)
- Printed final losses per method

---

### Q2 — Stock Open Price Forecasting

Notebook: `notebooks/cau2_stock_forecasting.ipynb`

Goal: forecast future **Open** prices using a sliding window.

Core preprocessing:
- Filter by `Ticker`
- Sort by `Date`
- Scale `Open` using **MinMaxScaler**
- Build sequences with `sequence_length = 60`
  - Input `X`: last 60 Open values
  - Target `y`: next Open value

Models covered in the notebook:
- **Linear Regression (neural form)**: single Dense layer output
- **Feedforward Neural Network (FFNN)**: Dense(50) → Dense(50) → Dense(1)
- **RNN/LSTM**: stacked LSTM layers (sequence input)
- **Decision Tree Regressor** baseline (scikit-learn)
- Regularization utilities: **Dropout**, **L2 Regularization**, **EarlyStopping** (used for NN variants)

Evaluation:
- **MSE** (Mean Squared Error)
- **R² score**
- Plot predicted vs actual (by date) for each ticker

Outputs:
- Metrics printed per ticker and per model
- Plots for actual vs predicted time series
- Training histories (loss curves) for neural nets

---

### Q3 — MNIST CNN Classification

Notebook: `notebooks/cau3_mnist_cnn.ipynb`

Goal: classify handwritten digits (0–9) with CNN.

Pipeline:
1. Load MNIST from `keras.datasets`
2. Normalize images to `[0, 1]`, reshape to `(28, 28, 1)`
3. One-hot encode labels

CNN architecture (from the notebook):
- Input `(28, 28, 1)`
- `Conv2D(32, 3×3, ReLU)` → `MaxPool(2×2)`
- `Conv2D(64, 3×3, ReLU)` → `MaxPool(2×2)`
- `Flatten`
- `Dense(128, ReLU)` → `Dropout(0.5)`
- `Dense(10, Softmax)`

Training:
- Optimizer: **Adam**
- Loss: `categorical_crossentropy`
- Batch size: `128`
- Epochs: `50`
- Includes validation tracking

Outputs:
- Accuracy & loss curves
- Test set evaluation
- Sample predictions visualization

---

## How to Run

### 1) Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start Jupyter

```bash
jupyter lab
```

Open notebooks in `notebooks/` and run cells from top to bottom.

---

## Reproducibility Notes

- For consistent results, keep random seeds fixed (some notebooks already include seeding).
- Neural networks may still show minor variance depending on:
  - CPU vs GPU
  - TensorFlow version
  - Non-deterministic GPU kernels

If you want fully deterministic runs, you can additionally configure TensorFlow deterministic ops (optional).

---

## Authors

- 52200207 – **Phạm Tuấn Đạt**
- 52200214 – **Trần Hồ Hoàng Vũ**
- 52200216 – **Trần Khiết Lôi**

---

## License

MIT (or update to your preferred license before publishing)
