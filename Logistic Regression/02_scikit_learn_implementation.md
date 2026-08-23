---
topic: "Logistic Regression"
subtopic: "From-Scratch Implementation (NumPy) & Scikit-Learn API, Hyperparameters & Solvers"
level: "Beginner/Intermediate"
doc_id: "logreg_02"
source_url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
scikit_learn_version: "1.4"
key_concepts:
  - "Gradient Descent theo lô & Vector hóa"
  - "Log Loss & Hàm kích hoạt Sigmoid"
  - "Scikit-Learn API & khả năng tương thích Solver"
  - "Tinh chỉnh siêu tham số (C, Penalty, Solver)"
  - "Pipeline thực chiến & Tiền xử lý"
---

# Cài Đặt Thuật Toán & Scikit-Learn API: Từ Tự Code (Scratch) Đến Thực Chiến (Production)

---

## 0. Cài Đặt Logistic Regression Từ Đầu Với NumPy

### 0.1. Vì sao phải "bẻ lái" tự code tay trước khi dùng thư viện?

Sử dụng `sklearn.linear_model.LogisticRegression` giúp bạn huấn luyện mô hình chỉ trong 2 dòng code. Tuy nhiên, cách này giấu đi toàn bộ bộ máy bên trong (black box).

Tự cài đặt thuật toán bằng **NumPy** giúp bạn:

1. **Hiểu bản chất Vector hóa (Vectorization):** Biến các vòng lặp toán học thành phép nhân ma trận tối ưu tốc độ.
2. **Cảm nhận sự hội tụ (Convergence):** Tận mắt thấy thuật toán Gradient Descent nhích từng bước để tìm trọng số tối ưu.
3. **Làm chủ phỏng vấn:** Sẵn sàng trả lời các câu hỏi Live Coding yêu cầu tự triển khai thuật toán không dùng thư viện ngoài.

---

### 0.2. Nhắc lại toán học & Ma trận hóa (Vectorization)

Theo tài liệu `logreg_01`, ta có các công thức cốt lõi:

**1. Dự đoán xác suất (Forward Pass):**

$$
z = \mathbf{X}\mathbf{w} + b, \quad \hat{\mathbf{y}} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $\mathbf{X}$: Ma trận dữ liệu kích thước $(N, D)$ với $N$ mẫu và $D$ đặc trưng.
- $\mathbf{w}$: Vector trọng số kích thước $(D, 1)$.
- $\hat{\mathbf{y}}$: Vector xác suất đầu ra kích thước $(N, 1)$.

**2. Đạo hàm Log Loss (Gradient Computation):**

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) \quad \text{(Shape: } (D, 1)\text{)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \quad \text{(Scalar)}
$$

**3. Cập nhật tham số (Gradient Descent Update):**

$$
\mathbf{w} := \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}, \qquad b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

---

### 0.3. Xây dựng Class `LogisticRegressionScratch`

```python
import numpy as np

class LogisticRegressionScratch:
    """
    Cài đặt Logistic Regression từ đầu bằng Batch Gradient Descent.
    Sử dụng hoàn toàn NumPy, bám sát công thức toán học cơ bản.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None    # Vector trọng số w, shape: (n_features,)
        self.bias = None       # Hệ số tự do b, scalar
        self.loss_history = [] # Lưu trữ loss để vẽ biểu đồ hội tụ

    def _sigmoid(self, z):
        # Tránh lỗi Tràn số (Overflow) e^{-z} khi z quá nhỏ (ví dụ z = -1000)
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _compute_loss(self, y_true, y_pred):
        # Thêm epsilon nhỏ để tránh lỗi log(0) gây ra kết quả NaN
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Bước 1: Khởi tạo trọng số w = 0 và b = 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Bước 2: Vòng lặp tối ưu hóa Gradient Descent
        for i in range(self.n_iterations):
            # 2.1 Forward pass: Tính z và xác suất dự đoán y_hat
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # 2.2 Backward pass: Tính Gradient (đạo hàm)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 2.3 Update pass: Cập nhật trọng số theo hướng ngược Gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 2.4 Lưu lại giá trị hàm mất mát (Loss)
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i:4d} | Log Loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        """Trả về xác suất thuộc về lớp 1: P(y=1|X)"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Cắt ngưỡng xác suất để đưa ra nhãn nhị phân 0 hoặc 1"""
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 0.4. Đọc hiểu chi tiết mã nguồn

| Dòng code tiêu biểu | Ý nghĩa toán học & Kỹ thuật lập trình |
| --- | --- |
| `np.clip(z, -500, 500)` | Tránh tình trạng văng lỗi `OverflowError: math range error` khi tính $e^{-z}$ với $z$ cực âm. |
| `np.clip(y_pred, 1e-15, 1 - 1e-15)` | Bảo vệ hàm `np.log()` khỏi giá trị $0$, vì $\ln(0) \to -\infty$ sẽ làm sập chương trình. |
| `dw = (1 / n) * np.dot(X.T, error)` | Nhân ma trận chuyển vị $\mathbf{X}^T$ có kích thước $(D, N)$ với vector sai số $(N, 1)$, tạo ra vector gradient $(D, 1)$ cực nhanh. |
| `self.weights -= lr * dw` | Bước nhảy Gradient Descent: Đi ngược chiều gradient để hạ thấp độ lỗi. |

### 0.5. Thực nghiệm đối chiếu: Code Scratch vs. Scikit-Learn

Để kiểm tra độ chính xác, ta so sánh kết quả của bản cài đặt tay với thư viện Scikit-Learn trên cùng một tập dữ liệu:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Khởi tạo dữ liệu giả lập
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=4,
    n_redundant=1, random_state=42
)

# 2. Phân chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu (Rất quan trọng với Gradient Descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện bản Scratch
scratch_model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
scratch_model.fit(X_train_scaled, y_train)
y_pred_scratch = scratch_model.predict(X_test_scaled)

# 5. Huấn luyện bản Scikit-Learn (Tắt penalty để đối chiếu công bằng)
sklearn_model = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
sklearn_model.fit(X_train_scaled, y_train)
y_pred_sklearn = sklearn_model.predict(X_test_scaled)

# 6. So sánh thông số
print("=== KẾT QUẢ ĐỐI CHIẾU ===")
print(f"Accuracy (Scratch)     : {accuracy_score(y_test, y_pred_scratch):.4f}")
print(f"Accuracy (Scikit-Learn): {accuracy_score(y_test, y_pred_sklearn):.4f}\n")

print("Trọng số w (Scratch)     :", np.round(scratch_model.weights, 4))
print("Trọng số w (Scikit-Learn):", np.round(sklearn_model.coef_[0], 4))
print(f"Bias b (Scratch)        : {scratch_model.bias:.4f}")
print(f"Bias b (Scikit-Learn)   : {sklearn_model.intercept_[0]:.4f}")
```

**Nhận xét kết quả:** Độ chính xác (Accuracy) và các hệ số trọng số $\mathbf{w}, b$ của 2 mô hình gần như khớp hoàn toàn. Điều này chứng minh thuật toán tự viết hoạt động hoàn toàn chính xác!

### 0.6. Trực quan hóa đường cong hội tụ (Loss Curve)

Đường cong Loss biểu thị lượng thông tin sai lệch giảm dần theo từng vòng lặp:

```python
plt.figure(figsize=(8, 4))
plt.plot(scratch_model.loss_history, color='blue', linewidth=2)
plt.title("Đường Cong Hội Tụ Của Gradient Descent (Loss Curve)")
plt.xlabel("Số Vòng Lặp (Iteration)")
plt.ylabel("Log Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

- **Đường cong dốc xuống mượt mà:** Thuật toán đang học tốt và giảm lỗi ổn định.
- **Đường cong đi ngang (Plateau):** Mô hình đã hội tụ, tăng thêm vòng lặp cũng không cải thiện thêm.
- **Nếu đường cong nhảy vọt lên:** Tốc độ học `learning_rate` bị đặt quá cao (Overshooting).

---

## 1. Khám Phá Scikit-Learn API (`sklearn.linear_model.LogisticRegression`)

Khi chuyển sang môi trường làm việc thực tế, ta sử dụng API chính thức của scikit-learn.

**Các phương thức cốt lõi cần nhớ:**

- `.fit(X, y)`: Tiếp nhận dữ liệu huấn luyện để giải toán tối ưu hóa.
- `.predict(X)`: Trả về nhãn dự đoán nhị phân ($0$ hoặc $1$).
- `.predict_proba(X)`: Trả về mảng 2 chiều chứa $[P(y=0), P(y=1)]$.
- `.decision_function(X)`: Trả về khoảng cách đại số $z = \mathbf{w}^T\mathbf{x} + b$. Khoảng cách này càng xa $0$, mô hình càng tự tin.

```python
# Ví dụ về sự khác biệt giữa predict và predict_proba
sample = X_test_scaled[:1]
print("Xác suất [P(y=0), P(y=1)] :", sklearn_model.predict_proba(sample))
print("Nhãn dự đoán cứng (0/1)   :", sklearn_model.predict(sample))
print("Giá trị đại số z          :", sklearn_model.decision_function(sample))
```

---

## 2. Ma Trận Siêu Tham Số (Hyperparameter Configuration)

Siêu tham số là những "núm vặn" điều chỉnh cách thuật toán học dữ liệu:

| Tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa thực tế & Ẩn dụ |
| --- | --- | --- | --- |
| `penalty` | str | `'l2'` | Loại phạt trọng số: `'l1'`, `'l2'`, `'elasticnet'`, `None`. |
| `C` | float | `1.0` | Sợi dây dắt chó: $C$ nhỏ → Phạt mạnh (giữ trọng số nhỏ, tránh Overfit). $C$ lớn → Phạt thả ga (dễ Overfit). |
| `solver` | str | `'lbfgs'` | Thuật toán giải toán tối ưu hóa (chi tiết ở Phần 3). |
| `max_iter` | int | `100` | Số bước chân tối đa cho phép Gradient Descent đi tìm đáy. Nên tăng lên $\ge 1000$. |
| `class_weight` | dict/str | `None` | Đặt `'balanced'` để hỗ trợ tập dữ liệu bị mất cân bằng nhãn nặng (ví dụ: 99% không bệnh, 1% có bệnh). |
| `tol` | float | `1e-4` | Ngưỡng kiên nhẫn. Nếu Loss cải thiện ít hơn `tol`, thuật toán lập tức dừng lại để tiết kiệm thời gian. |
| `multi_class` | str | `'auto'` | Chiến lược phân loại nhiều hơn 2 lớp: `'ovr'` (One-vs-Rest) hoặc `'multinomial'` (Softmax). |

---

## 3. Ma Trận Tương Thích Giữa Solvers & Penalty

Không phải thuật toán tối ưu nào (solver) cũng hỗ trợ tất cả các hình thức phạt (penalty). Chọn sai solver sẽ khiến mã nguồn văng lỗi!

**Ẩn dụ chọn Solver:**

- `liblinear`: Xe đạp — Nhẹ nhàng, dùng cho dữ liệu nhỏ.
- `lbfgs`: Xe sedan — Toàn diện, phù hợp cho dữ liệu vừa và nhỏ (Mặc định).
- `saga`: Tàu hỏa siêu tốc — Phù hợp cho Big Data, hỗ trợ mọi loại penalty.

**Bảng tra cứu khả năng tương thích:**

| Solver | L1 Penalty | L2 Penalty | ElasticNet | No Penalty | Trường hợp sử dụng tối ưu |
| --- | --- | --- | --- | --- | --- |
| `lbfgs` | ❌ | Có | ❌ | Có | Dữ liệu vừa và nhỏ. Thuật toán tối ưu mặc định chuẩn xác. |
| `liblinear` | Có | Có | ❌ | ❌ | Dữ liệu rất nhỏ. Thích hợp bài toán nhị phân. |
| `saga` | Có | Có | Có | Có | Toàn năng! Dữ liệu lớn, đặc trưng thưa (Sparse matrix). |
| `sag` | ❌ | Có | ❌ | Có | Dữ liệu rất nhiều mẫu ($N$ rất lớn). |
| `newton-cg` | ❌ | Có | ❌ | Có | Tính ma trận Hessian chính xác, yêu cầu lượng RAM lớn. |

---

## 4. Bài Toán Phân Loại Đa Lớp (Multi-Class Classification)

Khi cần phân loại $K > 2$ lớp (Ví dụ: Phân loại hoa thành 3 loài: Setosa, Versicolor, Virginica), Scikit-Learn cung cấp 2 phương pháp:

### 1. One-vs-Rest (OvR) — `multi_class='ovr'`

Huấn luyện $K$ mô hình nhị phân riêng biệt.

- Mô hình 1: Lớp A vs (Lớp B + Lớp C).
- Mô hình 2: Lớp B vs (Lớp A + Lớp C).
- Lớp nào có xác suất cao nhất sẽ được chọn làm kết quả cuối cùng.

### 2. Multinomial / Softmax — `multi_class='multinomial'`

Huấn luyện 1 mô hình duy nhất tối ưu đồng thời xác suất cho cả $K$ lớp bằng hàm Softmax:

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}
$$

---

## 5. Quy Trình Thực Chiến Chuẩn (Production-Ready Pipeline)

Trong dự án thực tế, không bao giờ truyền dữ liệu thô trực tiếp vào mô hình `LogisticRegression`. Ta phải đóng gói các bước Chuẩn hóa dữ liệu (Scaling) và Mô hình vào một `Pipeline`:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Tải tập dữ liệu Chẩn đoán Ung thư vú (Breast Cancer)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Phân chia Train/Test theo tỉ lệ 80/20 (Giữ nguyên tỉ lệ nhãn bằng stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Đóng gói Quy trình xử lý vào Pipeline
production_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Bước 1: Chuẩn hóa dữ liệu về N(0, 1)
    ('classifier', LogisticRegression(
        penalty='l2',             # Dùng điều hòa Ridge
        C=1.0,                    # Mức độ phạt trung bình
        solver='lbfgs',           # Thuật toán tối ưu hóa mặc định
        max_iter=1000,            # Tăng số vòng lặp tối đa
        class_weight='balanced',  # Tự cân bằng trọng số nhãn
        random_state=42
    ))
])

# 4. Huấn luyện toàn bộ Pipeline
production_pipeline.fit(X_train, y_train)

# 5. Đánh giá chất lượng
y_pred = production_pipeline.predict(X_test)
y_proba = production_pipeline.predict_proba(X_test)[:, 1]

print("=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print(f"Chỉ số ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nMa trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
```

---

## 6. Khung Dẫn Dắt Bài Tập & Dự Án (Exercise Framework)

Dựa vào file này, bạn có thể triển khai 3 dạng bài tập tự luyện:

### Dạng 1: Bài tập Lập trình Cơ bản (Code Completion)

- **Yêu cầu:** Hãy thêm L2 Regularization vào class `LogisticRegressionScratch`.
- **Gợi ý:** Sửa công thức tính gradient `dw` thành: `dw = (1/n) * np.dot(X.T, error) + (l2_param / n) * self.weights`.

### Dạng 2: Bài tập Thực nghiệm Siêu tham số (Hyperparameter Tuning)

- **Yêu cầu:** Viết vòng lặp thử nghiệm các giá trị $C \in [0.001, 0.01, 0.1, 1, 10, 100]$ trên tập dữ liệu bị Overfitting. Vẽ đồ thị biểu diễn Accuracy trên tập Train và Test để tìm ra điểm $C$ tối ưu.

### Dạng 3: Bài tập Xây dựng Pipeline Chống Rò Rỉ Dữ Liệu (Data Leakage)

- **Yêu cầu:** Giải thích tại sao việc gọi `StandardScaler().fit_transform(X)` trên toàn bộ dữ liệu trước khi `train_test_split` lại bị coi là lỗi rò rỉ dữ liệu. Hãy sửa lại bằng cách sử dụng `Pipeline`.

---

## 7. Tài Liệu Tham Khảo (Official References)

- Scikit-Learn Official User Guide: [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Andrew Ng, *Machine Learning Specialization* — Course 1: Supervised Machine Learning (Logistic Regression & Vectorization).
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.3.2.
