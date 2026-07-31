---
topic: "Logistic Regression"
subtopic: "Scikit-Learn API, Hyperparameters & Solvers"
level: "Intermediate"
doc_id: "logreg_02"
sources:
  - "Scikit-Learn v1.4 API Reference: sklearn.linear_model.LogisticRegression"
---

# Scikit-Learn API & Implementation Guide

## 1. Main Class Specification
Lớp chính được sử dụng trong Scikit-Learn là `sklearn.linear_model.LogisticRegression`. Mô hình cung cấp các phương thức cốt lõi:
* `.fit(X, y)`: Huấn luyện mô hình dựa trên dữ liệu đầu vào.
* `.predict(X)`: Trả về nhãn dự đoán ($0$ hoặc $1$).
* `.predict_proba(X)`: Trả về mảng xác suất cho từng lớp $[P(y=0), P(y=1)]$.
* `.decision_function(X)`: Trả về khoảng cách từ mẫu dữ liệu tới siêu phẳng quyết định ($z = \mathbf{w}^T \mathbf{x} + b$).

## 2. Hyperparameter Configuration Matrix

| Tham số | Kiểu dữ liệu | Mặc định | Mức độ ảnh hưởng & Hướng dẫn sử dụng |
| :--- | :--- | :--- | :--- |
| `penalty` | `str` | `'l2'` | Loại điều hòa: `'l1'`, `'l2'`, `'elasticnet'`, `None`. Quyết định cách phạt trọng số lớn. |
| `C` | `float` | `1.0` | Hệ số nghịch đảo điều hòa ($C > 0$). $C$ nhỏ $\rightarrow$ Phạt mạnh (ngừa Overfitting). $C$ lớn $\rightarrow$ Phạt yếu (ngừa Underfitting). |
| `solver` | `str` | `'lbfgs'` | Thuật toán tối ưu hóa loss function: `'lbfgs'`, `'liblinear'`, `'saga'`, `'sag'`, `'newton-cg'`. |
| `max_iter` | `int` | `100` | Số lượng vòng lặp tối đa để thuật toán tối ưu hội tụ. Nên đặt $\ge 1000$ trong thực tế. |
| `class_weight`| `dict` / `str`| `None` | Xử lý mất cân bằng nhãn. Đặt `'balanced'` để tự động tính trọng số tỉ lệ nghịch với tần suất lớp. |
| `tol` | `float` | `1e-4` | Ngưỡng dừng tối ưu (Tolerance for stopping criteria). Khi mức thay đổi loss $< \text{tol}$, thuật toán dừng. |
| `multi_class` | `str` | `'auto'` | Chiến lược phân loại đa lớp: `'ovr'` (One-vs-Rest) hoặc `'multinomial'` (Softmax Regression). |
| `random_state`| `int` | `None` | Cố định hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả (Reproducibility). |
| `l1_ratio` | `float` | `None` | Tỉ lệ điều chỉnh L1 khi `penalty='elasticnet'`. Giá trị nằm trong khoảng $[0, 1]$. |

## 3. Solver Compatibility Matrix
Việc chọn thuật toán tối ưu (`solver`) phụ thuộc vào kích thước dữ liệu và loại `penalty` được chọn:

| Solver | L1 Penalty | L2 Penalty | ElasticNet | No Penalty | Quy mô dữ liệu phù hợp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `'lbfgs'` | Không | **Có** | Không | **Có** | Dữ liệu nhỏ đến vừa (Mặc định tối ưu cho đa số bài toán). |
| `'liblinear'` | **Có** | **Có** | Không | Không | Dữ liệu rất nhỏ. Không hỗ trợ tối ưu đồng thời Multinomial. |
| `'saga'` | **Có** | **Có** | **Có** | **Có** | Dữ liệu lớn, thuộc tính thưa (sparse). Hỗ trợ toàn bộ penalty. |
| `'sag'` | Không | **Có** | Không | **Có** | Dữ liệu rất lớn, hội tụ nhanh nhờ Stochastic Average Gradient. |
| `'newton-cg'` | Không | **Có** | Không | **Có** | Dữ liệu vừa, tính ma trận Hessian chính xác nhưng tốn RAM. |

## 4. Multi-Class Strategies Overview
Khi biến mục tiêu có $K > 2$ lớp phân biệt:

1. **One-vs-Rest (OvR)**: Bật bằng `multi_class='ovr'`. Tạo ra $K$ mô hình phân loại nhị phân độc lập. Mỗi mô hình phân tách lớp $k$ với tất cả các lớp còn lại.
2. **Multinomial (Softmax)**: Bật bằng `multi_class='multinomial'`. Tối ưu đồng thời hàm Cross-Entropy trên toàn bộ $K$ lớp bằng hàm Softmax:
   $$P(y=k\vert{}\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

## 5. Complete Production Executable Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Khởi tạo dữ liệu mẫu ngẫu nhiên
X_raw, y_raw = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2, 
    random_state=42
)

# 2. Phân chia tập huấn luyện và kiểm thử (Phân tầng nhãn Stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# 3. Định nghĩa Pipeline đóng gói chuẩn
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])

# 4. Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# 5. Dự đoán và Đánh giá
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
```

## 6. Official References

Scikit-Learn Official Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
