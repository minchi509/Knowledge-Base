---
topic: "Logistic Regression"
subtopic: "From-Scratch Implementation (NumPy) & Scikit-Learn API, Hyperparameters & Solvers"
level: "Beginner/Intermediate"
doc_id: "logreg_02"
sources:
  - "Tự triển khai dựa trên công thức toán học ở tài liệu logreg_01"
  - "Scikit-Learn v1.4 API Reference: sklearn.linear_model.LogisticRegression"
---

# Cài Đặt Thuật Toán & Scikit-Learn API Implementation Guide

## 0. Cài Đặt Logistic Regression Từ Đầu (Không Dùng Thư Viện)

### 0.1. Vì sao cần cài đặt tay trước khi dùng thư viện?

Ở phần dưới đây, chúng ta sẽ dùng `sklearn.linear_model.LogisticRegression` để huấn luyện mô hình chỉ với vài dòng code. Điều này rất tiện cho công việc thực tế, nhưng lại **che giấu** toàn bộ phần "bên trong" của thuật toán — thứ mà người mới học cần hiểu rõ trước khi dùng thư viện.

Mục này sẽ cài đặt lại toàn bộ Logistic Regression **chỉ bằng NumPy**, bám sát đúng công thức toán đã trình bày ở tài liệu `logreg_01` (Sigmoid, Log Loss, đạo hàm, và Gradient Descent).

### 0.2. Nhắc lại công thức toán (tham chiếu `logreg_01`)

**Sigmoid:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b
$$

**Hàm mất mát Log Loss:**

$$
J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]
$$

**Đạo hàm (Gradient)** — đây là phần mà thư viện sklearn tự động tính giúp ta:

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}), \qquad
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

**Quy tắc cập nhật Gradient Descent** (với tốc độ học $\alpha$):

$$
\mathbf{w} := \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}, \qquad b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

Lặp lại quá trình này `n_iterations` lần, $\mathbf{w}$ và $b$ sẽ dần tiến tới giá trị tối ưu.

### 0.3. Cài đặt Class `LogisticRegressionScratch`

```python
import numpy as np

class LogisticRegressionScratch:
    """
    Cài đặt Logistic Regression từ đầu bằng Batch Gradient Descent.
    Không sử dụng bất kỳ hàm nào từ scikit-learn.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None   # vector w, shape (n_features,)
        self.bias = None      # scalar b
        self.loss_history = []

    def _sigmoid(self, z):
        # Giới hạn z để tránh tràn số (overflow) khi tính e^{-z}
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        # Thêm epsilon nhỏ để tránh log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Bước 1: Khởi tạo w = 0 (vector) và b = 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Bước 2: Vòng lặp Gradient Descent
        for i in range(self.n_iterations):
            # 2.1 Tính z và y_hat (forward pass)
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # 2.2 Tính đạo hàm (gradient)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 2.3 Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 2.4 (Tuỳ chọn) Lưu lại loss để vẽ biểu đồ hội tụ
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"Vòng lặp {i:4d} | Loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

**Giải thích từng bước cho người mới:**

| Bước trong code | Ý nghĩa toán học |
| :--- | :--- |
| `self.weights = np.zeros(n_features)` | Khởi tạo $\mathbf{w} = \mathbf{0}$, $b = 0$ — điểm xuất phát của Gradient Descent |
| `z = np.dot(X, self.weights) + self.bias` | Tính $z = \mathbf{w}^T\mathbf{x} + b$ cho toàn bộ tập dữ liệu (vector hoá bằng NumPy thay vì vòng lặp `for`) |
| `y_pred = self._sigmoid(z)` | Áp dụng $\sigma(z)$ để ra xác suất $\hat{y}$ |
| `dw = (1/n) * X.T @ (y_pred - y)` | Chính là công thức đạo hàm $\frac{\partial J}{\partial \mathbf{w}}$ ở trên |
| `self.weights -= learning_rate * dw` | Bước cập nhật Gradient Descent |

**Lưu ý quan trọng**: Đây là bản cài đặt cơ bản (educational) — **chưa có** Regularization (L1/L2), chưa hỗ trợ `solver` nâng cao như `lbfgs`/`saga`, và mặc định dùng Batch Gradient Descent (tính gradient trên toàn bộ tập dữ liệu mỗi vòng lặp). Đây chính là lý do vì sao thực tế ta luôn dùng thư viện sklearn — nó nhanh, ổn định và đầy đủ tính năng hơn nhiều.

### 0.4. Đối chiếu kết quả: Scratch vs. Scikit-Learn

Để chứng minh cài đặt trên là đúng, ta huấn luyện song song 2 mô hình trên cùng dữ liệu và so sánh:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Tạo dữ liệu mẫu
X, y = make_classification(
    n_samples=500, n_features=5, n_informative=4,
    n_redundant=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Chuẩn hóa dữ liệu (BẮT BUỘC với Gradient Descent tự cài đặt,
#    vì code trên không có cơ chế tối ưu solver nâng cao như lbfgs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Huấn luyện bản Scratch
model_scratch = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
model_scratch.fit(X_train_scaled, y_train)
y_pred_scratch = model_scratch.predict(X_test_scaled)

# 4. Huấn luyện bản Scikit-Learn để đối chiếu
model_sklearn = LogisticRegression(max_iter=1000, random_state=42)
model_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = model_sklearn.predict(X_test_scaled)

# 5. So sánh
print("=== SO SÁNH KẾT QUẢ ===")
print(f"Accuracy (Scratch)      : {accuracy_score(y_test, y_pred_scratch):.4f}")
print(f"Accuracy (Scikit-Learn) : {accuracy_score(y_test, y_pred_sklearn):.4f}")

print("\nTrọng số w (Scratch)      :", np.round(model_scratch.weights, 3))
print("Trọng số w (Scikit-Learn) :", np.round(model_sklearn.coef_[0], 3))
print(f"Bias b (Scratch)      : {model_scratch.bias:.3f}")
print(f"Bias b (Scikit-Learn) : {model_sklearn.intercept_[0]:.3f}")
```

**Kỳ vọng đầu ra**: Với dữ liệu đã chuẩn hóa và đủ số vòng lặp, `accuracy` của 2 mô hình sẽ xấp xỉ bằng nhau, và các trọng số $\mathbf{w}$, $b$ cũng xấp xỉ nhau (không tuyệt đối giống hệt vì sklearn mặc định có Regularization L2 với $C=1.0$, còn bản Scratch thì không).

### 0.5. Trực quan hóa quá trình hội tụ (Loss Curve)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(model_scratch.loss_history)
plt.xlabel("Số vòng lặp (Iteration)")
plt.ylabel("Log Loss")
plt.title("Quá trình hội tụ của Gradient Descent")
plt.grid(True, alpha=0.3)
plt.show()
```

Nếu đường cong Loss giảm dần và đi ngang (plateau) ở cuối, mô hình đã hội tụ tốt. Nếu Loss dao động mạnh hoặc tăng lên, khả năng cao `learning_rate` đang đặt quá lớn — đây là bài học thực hành trực quan giúp người mới hiểu vai trò của siêu tham số `learning_rate`.

### 0.6. Bài tập đề xuất cho người học (không bắt buộc)

1. Thử đổi `learning_rate` thành `10` (quá lớn) và `0.0001` (quá nhỏ), quan sát `loss_history` để thấy hiện tượng "vượt quá điểm tối ưu" (overshooting) và "hội tụ quá chậm".
2. Thêm **L2 Regularization** vào công thức gradient (gợi ý: cộng thêm $\frac{\lambda}{N}\mathbf{w}$ vào `dw`) rồi so sánh lại với sklearn khi đặt `penalty='l2'`.
3. Thay Batch Gradient Descent bằng **Stochastic Gradient Descent (SGD)** — cập nhật tham số sau mỗi 1 mẫu dữ liệu thay vì toàn bộ tập.

Sau khi hiểu bản Scratch này, phần tiếp theo sẽ giới thiệu Scikit-Learn API — về bản chất, mọi tham số như `solver`, `max_iter`, `tol`, `C` chỉ là các "phiên bản nâng cao, tối ưu hơn" của chính vòng lặp Gradient Descent vừa cài đặt thủ công ở trên.

---

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

Tham khảo thêm cho phần cài đặt thủ công (mục 0):
- Andrew Ng, *Machine Learning Specialization* — Course 1: Supervised Machine Learning, Module về Logistic Regression.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.3.2.
