
# Parkinson’s Disease Classification Repository  
*A Comparative Study of Classical and Fully-Quantum Machine Learning Models*

---

## 📖 Abstract
Parkinson’s disease (PD) is a neurodegenerative disorder caused by progressive dopamine neuron loss, leading to motor and speech impairments.  
This repository provides a **scientific and reproducible framework** for comparing **classical ML models** and a **fully quantum variational classifier** using the [UCI Parkinson’s dataset](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification).

---

## 📂 Repository Structure
```

.
├── data/
│   ├── raw/               # Raw dataset (downloaded automatically, not committed)
│   └── processed/         # Preprocessed train/test data
├── notebooks/
│   ├── 01\_preprocessing.ipynb
│   ├── 02\_classical\_training.ipynb
│   ├── 03\_classical\_testing.ipynb
│   ├── 04\_quantum\_training.ipynb
│   └── 05\_quantum\_testing.ipynb
├── scripts/
│   └── download\_data.py
├── models/                # Saved models + scalers
├── results/               # Metrics, plots, logs
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/parkinsons-quantum-vs-classical.git
cd parkinsons-quantum-vs-classical
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
````

---

## 📊 Workflow

1. `python scripts/download_data.py` → fetch raw dataset
2. Run `notebooks/01_preprocessing.ipynb` → create train/test sets
3. Run `notebooks/02_classical_training.ipynb` → train classical models
4. Run `notebooks/03_classical_testing.ipynb` → evaluate classical models
5. Run `notebooks/04_quantum_training.ipynb` → train fully quantum model
6. Run `notebooks/05_quantum_testing.ipynb` → evaluate on simulator & QPU

---

## 📈 Evaluation Metrics

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* Quantum-specific: **circuit depth**, **number of qubits**, **parameters**
* Robustness under **noisy simulation**
* Final inference on **real QPU hardware**

---

## 📑 Files


### `requirements.txt`

```txt
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.9.2
joblib==1.4.2
qiskit==1.2.4
pennylane==0.38.0
jupyterlab==4.2.5
notebook==7.2.2
```

### `scripts/download_data.py`

```python
"""
Download the Parkinson's dataset from UCI Repository.
"""

import os
import urllib.request
import hashlib
from datetime import datetime

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00345/Parkinsons.csv"
DATA_DIR = "data/raw"
FILE_NAME = "parkinsons.csv"

def sha256_checksum(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, FILE_NAME)

    if not os.path.exists(file_path):
        print(f"[{datetime.now()}] Downloading dataset...")
        urllib.request.urlretrieve(URL, file_path)
        print(f"Dataset saved to {file_path}")
    else:
        print("Dataset already exists. Skipping download.")

    checksum = sha256_checksum(file_path)
    print(f"SHA-256 checksum: {checksum}")

if __name__ == "__main__":
    download_dataset()
```

### `notebooks/01_preprocessing.ipynb`

```
# Code
```

---

## 📌 Notes

* All random splits fixed (`random_state=42`) for reproducibility.

```
