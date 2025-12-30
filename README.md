# Intro to Machine Learning – Final Project

Final project (Ton Duc Thang University, 2024) covering:
- **Q1:** Optimization methods (Gradient Descent, Momentum, Adagrad) + experiments  
- **Q2:** Stock **Open** price forecasting (sequence length = 60) with Neural Nets (FFNN/RNN/LSTM) and baseline ML models  
- **Q3:** MNIST classification using CNN

---

## Repository Structure

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

---

## Datasets
- `HousingData.csv` – housing regression dataset (CSV)  
- `data_src_2.csv` – combined dataset (includes stock OHLCV + metadata like Industry, GDP)

---

## Setup
> Recommended: Python 3.10+

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run
Open notebooks:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Authors
- 52200207 – Phạm Tuấn Đạt  
- 52200214 – Trần Hồ Hoàng Vũ  
- 52200216 – Trần Khiết Lôi  

---

## License
MIT (or choose your preferred license)
