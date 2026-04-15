# ECE 4363 / ECE 5363 — Pattern Recognition  
## Project 4: Gaussian Bayes Classification  

**Filename:** `Lastname_Project4.py`

---

## 1. Overview

This project implements and compares three Gaussian Bayes classifiers on a synthesized two-class, five-dimensional problem.

Each classifier assigns a sample $x \in \mathbb{R}^5$ to the class:

$$
k^* = \arg\max_k \left[ \log p(x \mid k) + \log P(k) \right]
$$

- $p(x \mid k)$: multivariate Gaussian density  
- $P(k) = 0.5$ for both classes  

The classifiers differ only in how their parameters are obtained.

---

## 2. Problem Setup

### 2.1 Class Distributions

Both classes follow multivariate Gaussian distributions:

### Class 1

$$
m_1 = [0, 0, 0, 0, 0]^T
$$

$$
S_1 =
\begin{bmatrix}
0.80 & 0.20 & 0.10 & 0.05 & 0.01 \\
0.20 & 0.70 & 0.10 & 0.03 & 0.02 \\
0.10 & 0.10 & 0.80 & 0.02 & 0.01 \\
0.05 & 0.03 & 0.02 & 0.90 & 0.01 \\
0.01 & 0.02 & 0.01 & 0.01 & 0.80
\end{bmatrix}
$$

### Class 2

$$
m_2 = [1, 1, 1, 1, 1]^T
$$

$$
S_2 =
\begin{bmatrix}
0.90 & 0.10 & 0.05 & 0.02 & 0.01 \\
0.10 & 0.80 & 0.10 & 0.02 & 0.02 \\
0.05 & 0.10 & 0.70 & 0.02 & 0.01 \\
0.02 & 0.02 & 0.02 & 0.60 & 0.02 \\
0.01 & 0.02 & 0.01 & 0.02 & 0.70
\end{bmatrix}
$$

**Priors:**

$$
P(C_1) = P(C_2) = 0.5
$$

---

### 2.2 Dataset Sizes

| Experiment | N_train | N_test |
|----------|--------|--------|
| Round 1  | 100    | 10,000 |
| Round 2  | 1000   | 10,000 |

- Test set is fixed across both rounds  
- Data can be:
  - Loaded from Excel
  - Generated using NumPy with seeds:
    - Training: `0`
    - Test: `100`

---

## 3. Classifiers

All classifiers use the same discriminant function:

$$
g_k(x) =
-0.5 (x - m_k)^T S_k^{-1} (x - m_k)
- 0.5 \log |S_k|
+ \log P(k)
$$

Decision rule:

$$
k^* = \arg\max_k g_k(x)
$$

---

### 3.1 True Bayes

- Uses known parameters $m_k$, $S_k$  
- No training data required  
- Achieves **Bayes optimal error (lower bound)**  

---

### 3.2 MLE Bayes

Parameters estimated from training data:

$$
m_k = \frac{1}{n_k} \sum_{i=1}^{n_k} x_i
$$

$$
S_k = \frac{1}{n_k - 1} \sum_{i=1}^{n_k} (x_i - m_k)(x_i - m_k)^T
$$

- Uses full covariance matrix  
- Converges to True Bayes as $N_{train} \to \infty$  

---

### 3.3 Naive Bayes

Covariance restricted to diagonal:

$$
S_k^{\text{naive}} =
\text{diag}(\text{var}(x^1|k), \dots, \text{var}(x^5|k))
$$

- Assumes feature independence  
- Violated in this dataset (off-diagonal terms up to 0.20)  
- Expected to have highest error  

---

## 4. Implementation

### 4.1 Class: `BayesClassifier`

**Methods:**

- `__init__` — stores true parameters  
- `generate_data()` — synthetic data generation  
- `load_data()` — Excel loading  
- `fit(X, y)` — parameter estimation  
- `_log_likelihood()` — discriminant core  
- `predict_true()` — true parameters  
- `predict_mle()` — MLE parameters  
- `predict_naive()` — diagonal covariance  
- `error_rate()` — misclassification rate  
- `plot_histogram()` — visualization  

---

### 4.2 Dependencies

- `numpy` — linear algebra, sampling  
- `pandas` — Excel I/O  
- `matplotlib` — visualization  

---

### 4.3 Usage

```bash
python Lastname_Project4.py
```

- Runs both experiments  
- Outputs results table  

To load Excel data:

```python
load_data(train_file, test_file)
```

---

## 5. Expected Results

$$
E_{\text{true}} \leq E_{\text{mle}} \leq E_{\text{naive}}
$$

- **True Bayes**: optimal baseline  
- **MLE Bayes**: improves with more data  
- **Naive Bayes**: biased due to independence assumption  

As $N_{train}$ increases:

- MLE error ↓  
- Naive error ↓  
- True error = constant  

---

## 6. File Structure

- `Lastname_Project4.py` — main submission  
- `main.py` — entry point  
- `train_N.xlsx` — training data  
- `test_10000.xlsx` — test data  

---

## 7. Notes

- Use `numpy.linalg.slogdet` for numerical stability  
- Use `numpy.einsum` for vectorized Mahalanobis computation  
- Convert labels from `(1,2)` → `(0,1)` if needed  

Call histogram method:

```python
BayesClassifier.plot_histogram(X, y)
```
