---
topic: "Logistic Regression"
subtopic: "Common Pitfalls, Troubleshooting & Best Practices"
level: "Intermediate/Advanced"
doc_id: "logreg_05"
source_url: "https://scikit-learn.org/stable/common_pitfalls.html"
scikit_learn_version: "1.4"
key_concepts:
  - "ConvergenceWarning & Cách khắc phục tối ưu hóa"
  - "Phòng tránh rò rỉ dữ liệu với Pipeline"
  - "Đa cộng tuyến & Hệ số phóng đại phương sai (VIF)"
  - "Tự tính VIF & minh họa rò rỉ dữ liệu bằng số liệu"
  - "Danh mục kiểm tra sẵn sàng triển khai"
---

# Các Lỗi Thường Gặp, Cách Khắc Phục & Quy Chuẩn Thực Chiến Trong Logistic Regression

---

## 1. Xử Lý Cảnh Báo Không Hội Tụ (ConvergenceWarning)

### 1.1. Hiện tượng

Khi huấn luyện mô hình `LogisticRegression`, thuật toán dừng lại và hiển thị thông báo cảnh báo đỏ:

```
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
```

### 1.2. Nguyên nhân gốc rễ

- **Dữ liệu chưa được chuẩn hóa thang đo:** Khi các thuộc tính chênh lệch thang đo lớn (ví dụ: Tuổi $[18, 80]$ và Thu nhập $[10^6, 10^8]$), mặt không gian hàm mất mát bị kéo giãn biến dạng. Các bước nhảy Gradient bị văng ra ngoài phạm vi hội tụ.
- **Số vòng lặp `max_iter` quá thấp:** Mặc định `max_iter=100` của Scikit-Learn không đủ để giải thuật tối ưu L-BFGS đạt tới sai số dừng `tol`.
- **Đa cộng tuyến nghiêm trọng (Multicollinearity):** Các đặc trưng có sự phụ thuộc tuyến tính cao làm ma trận Hessian gần như suy biến, khiến thuật toán tối ưu bị quẩn.
- **Hệ số điều hòa quá yếu:** Tham số $C$ quá lớn ($C > 10000$) triệt tiêu hiệu ứng phạt, làm bài toán khó giải.

### 1.3. Quy trình khắc phục chuẩn (Resolution Protocol)

1. **Bước 1:** Bắt buộc chuẩn hóa dữ liệu qua `StandardScaler()` trước khi đưa vào mô hình.
2. **Bước 2:** Tăng số lượng vòng lặp tối đa: `LogisticRegression(max_iter=1000)`.
3. **Bước 3:** Chuyển đổi solver sang `saga` hoặc `liblinear` nếu tập dữ liệu rất lớn hoặc có cấu trúc ma trận thưa (Sparse Matrix).

---

## 2. Phòng Tránh Rò Rỉ Dữ Liệu (Data Leakage)

### 2.1. Sai lầm phổ biến

Thực hiện thao tác tiền xử lý dữ liệu (như `fit_transform()` của `StandardScaler` hoặc `SimpleImputer`) trên toàn bộ dữ liệu trước khi gọi hàm `train_test_split()`.

**❌ QUY TRÌNH SAI (RÒ RỈ DỮ LIỆU):**

```
[Toàn bộ Dữ liệu] ---> fit_transform(StandardScaler) ---> train_test_split() ---> Train Model
```

**Hậu quả:** Giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ của tập kiểm thử (Test Set) bị tính gộp vào tập huấn luyện. Mô hình "học lén" thông tin của dữ liệu tương lai, dẫn đến điểm đánh giá giả tạo rất cao nhưng thất bại khi chạy thực tế (Production).

### 2.2. Giải pháp chuẩn với Pipeline

Thực hiện `fit` duy nhất trên tập Train, sau đó dùng các tham số đã học để `transform` cho tập Test. Cách an toàn nhất là đóng gói luồng vào `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Đóng gói tiền xử lý và mô hình vào Pipeline an toàn
secure_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Cross-Validation tự động cách ly từng Fold, tuyệt đối không bị Leakage
scores = cross_val_score(
    secure_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='f1'
)
```

### 2.3. Minh họa bản chất Data Leakage bằng tính toán số liệu

Mã nguồn dưới đây tự tính toán thủ công $\mu$ và $\sigma$ để làm rõ sự sai lệch giữa việc tính đúng và tính bị rò rỉ:

```python
import numpy as np

# Tạo dữ liệu giả lập
np.random.seed(42)
X_train = np.random.normal(loc=50, scale=10, size=80)  # 80 mẫu Train
X_test = np.random.normal(loc=70, scale=15, size=20)    # 20 mẫu Test (phân phối khác)

# ---- CÁCH SAI: Tính mu, sigma trên TOÀN BỘ dữ liệu (Train + Test) ----
X_all = np.concatenate([X_train, X_test])
mu_leak = X_all.mean()
sigma_leak = X_all.std()
X_train_leak = (X_train - mu_leak) / sigma_leak

# ---- CÁCH ĐÚNG: Tính mu, sigma CHỈ trên tập Train ----
mu_correct = X_train.mean()
sigma_correct = X_train.std()
X_train_correct = (X_train - mu_correct) / sigma_correct

print(f"mu tính ĐÚNG (chỉ Train)   : {mu_correct:.3f}")
print(f"mu bị RÒ RỈ (Train + Test) : {mu_leak:.3f}")
print(f"Sai lệch do Leakage        : {abs(mu_correct - mu_leak):.3f}")
```

---

## 3. Phát Hiện Đa Cộng Tuyến Bằng Chỉ Số VIF (Variance Inflation Factor)

### 3.1. Bản chất vấn đề

Hiện tượng đa cộng tuyến (các thuộc tính độc lập tương quan mạnh với nhau) không ảnh hưởng tới khả năng dự đoán toàn cục, nhưng làm bùng nổ phương sai của các hệ số trọng số $\mathbf{w}$. Điều này làm mất khả năng giải thích tầm quan trọng của từng thuộc tính trong mô hình.

### 3.2. Công thức VIF

Chỉ số phóng đại phương sai VIF đo lường mức độ gia tăng phương sai của hệ số ước lượng:

$$
\text{VIF}_i = \frac{1}{1 - R_i^2}
$$

Trong đó $R_i^2$ là hệ số xác định (Coefficient of Determination) thu được khi thực hiện hồi quy tuyến tính biến $x_i$ theo tất cả các biến độc lập còn lại.

| Giá trị VIF | Đánh giá mức độ đa cộng tuyến | Hành động xử lý |
| --- | --- | --- |
| $\text{VIF} = 1$ | Không có tương quan. | Giữ nguyên. |
| $1 < \text{VIF} < 5$ | Tương quan nhẹ/vừa phải (An toàn). | Giữ nguyên. |
| $\text{VIF} > 10$ | Đa cộng tuyến nghiêm trọng. | Loại bỏ bớt đặc trưng hoặc bật L2 Regularization ($C$ nhỏ). |

### 3.3. Cài đặt kiểm tra VIF thực tế và tự cài đặt thủ công (Scratch)

**Sử dụng `statsmodels` thực tế:**

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(X_df):
    """
    Tính toán chỉ số VIF cho dataframe đầu vào bằng statsmodels
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(X_df.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)
```

**Tự cài đặt VIF bằng Phương Trình Chuẩn (Normal Equation) trong NumPy:**

Để hiểu cách tính $R_i^2$, ta tự lập trình hồi quy bằng công thức $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$:

```python
import numpy as np
import pandas as pd

def r_squared_scratch(X_others, y_feature):
    """
    Tự hồi quy tuyến tính (Normal Equation) biến x_i theo các biến còn lại
    """
    n = X_others.shape[0]
    X_with_bias = np.column_stack([np.ones(n), X_others])

    # Phương trình chuẩn: beta = (X^T X)^-1 X^T y (dùng pinv để tránh ma trận suy biến)
    XtX = X_with_bias.T @ X_with_bias
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X_with_bias.T @ y_feature

    y_pred = X_with_bias @ beta
    ss_res = np.sum((y_feature - y_pred) ** 2)            # Tổng bình phương phần dư
    ss_tot = np.sum((y_feature - y_feature.mean()) ** 2)  # Tổng bình phương toàn phần

    r2 = 1 - ss_res / (ss_tot + 1e-15)
    return r2


def check_multicollinearity_scratch(X_df):
    """
    Tự tính VIF cho từng cột: VIF_i = 1 / (1 - R_i^2)
    """
    results = []
    columns = X_df.columns
    X_values = X_df.values

    for i, col in enumerate(columns):
        y_feature = X_values[:, i]
        X_others = np.delete(X_values, i, axis=1)

        r2_i = r_squared_scratch(X_others, y_feature)
        vif_i = 1 / (1 - r2_i + 1e-15)
        results.append({"Feature": col, "VIF": vif_i})

    return pd.DataFrame(results).sort_values(by="VIF", ascending=False)
```

---

## 4. Bảng Tra Cứu Nhanh Lỗi & Giải Pháp (Fast Troubleshooting Guide)

| Hiện tượng / Triệu chứng | Nguyên nhân chính | Giải pháp khắc phục nhanh |
| --- | --- | --- |
| `ConvergenceWarning` | Thiếu Feature Scaling hoặc `max_iter` nhỏ. | Thêm `StandardScaler()` và tăng `max_iter=1000`. |
| Test Accuracy cao bất thường nhưng triển khai bị sụt giảm | Data Leakage do `fit_transform()` trước split. | Đóng gói tiền xử lý vào `sklearn.pipeline.Pipeline`. |
| Trọng số $\mathbf{w}$ biến động rất lớn khi thay đổi nhẹ dữ liệu | Hiện tượng đa cộng tuyến ($\text{VIF} > 10$). | Loại bỏ thuộc tính trùng lặp hoặc dùng phạt L2 (`penalty='l2'`). |
| Mô hình dự đoán $100\%$ thuộc về một lớp nhãn | Mất cân bằng nhãn nặng (Class Imbalance). | Bật `class_weight='balanced'` và điều chỉnh Decision Threshold. |

---

## 5. Danh Mục Kiểm Tra Sẵn Sàng Triển Khai (Production Readiness Checklist)

Trước khi đóng gói file mô hình (`.joblib` hoặc `.pkl`) để triển khai trên REST API (FastAPI, Flask) hoặc ứng dụng thực tế, hãy kiểm tra các mục sau:

- [ ] Tất cả đặc trưng số đều đã đi qua `StandardScaler`.
- [ ] Tất cả đặc trưng phân loại đã áp dụng `OneHotEncoder(drop='first')` để loại bỏ bẫy biến giả.
- [ ] Đã ấn định tham số `max_iter` tối thiểu $1000$ để đảm bảo hội tụ.
- [ ] Cố định tham số `random_state` để đảm bảo tính tái lập kết quả (Reproducibility).
- [ ] Đã tính chỉ số VIF và không còn cột đặc trưng nào có $\text{VIF} > 10$.
- [ ] Toàn bộ luồng tiền xử lý và mô hình được đóng gói thống nhất trong một `Pipeline`.
- [ ] Đã kiểm thử hiệu năng trên tập dữ liệu hoàn toàn độc lập (Unseen Test Set).

---

## 6. Khung Dẫn Dắt Bài Tập & Thực Hành (Exercise Framework)

### Dạng 1: Sửa Lỗi Data Leakage

- **Yêu cầu:** Cho một đoạn mã Python bị rò rỉ dữ liệu khi thực hiện Imputation và Scaling. Hãy viết lại mã nguồn bằng cách sử dụng `sklearn.pipeline.Pipeline` và `ColumnTransformer`.

### Dạng 2: Lọc Đặc Trưng Dựa Trên VIF

- **Yêu cầu:** Viết một hàm `remove_high_vif_features(X_df, threshold=10.0)` tự động lặp lại quá trình tính VIF và xóa từng đặc trưng có VIF cao nhất cho đến khi tất cả các đặc trưng đều có $\text{VIF} \le 10$.

---

## 7. Tài Liệu Tham Khảo (Official References)

- Scikit-Learn Official User Guide: [Common Pitfalls & Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html)
- Statsmodels Documentation: Variance Inflation Factor
