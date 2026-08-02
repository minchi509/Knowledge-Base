---
topic: "Logistic Regression"
subtopic: "Common Pitfalls, Troubleshooting & Best Practices (kèm minh họa/cài đặt thủ công)"
level: "Intermediate/Advanced"
doc_id: "logreg_05"
sources:
  - "Tự triển khai minh họa Data Leakage và VIF dựa trên công thức toán ở chính tài liệu này"
  - "Scikit-Learn Official User Guide: Common Pitfalls"
---

# Common Pitfalls, Troubleshooting & Best Practices

## 1. Troubleshooting `ConvergenceWarning`

### Problem Statement

Khi huấn luyện mô hình `LogisticRegression`, xuất hiện cảnh báo đỏ:

`ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.`

### Underlying Causes

1. **Dữ liệu chưa được chuẩn hóa**: Các thuộc tính nằm trên các thang đo khác nhau làm cho bề mặt hàm mất mát bị kéo giãn, làm bước nhảy gradient bị văng ra ngoài phạm vi hội tụ.

2. **Số vòng lặp `max_iter` quá thấp**: Giá trị mặc định `max_iter=100` không đủ để giải thuật L-BFGS đạt tới ngưỡng dừng `tol`.

3. **Đa cộng tuyến nghiêm trọng**: Dữ liệu có các thuộc tính phụ thuộc tuyến tính lẫn nhau.

4. **Hệ số điều hòa quá yếu**: Tham số $C$ được đặt quá lớn ($C > 10000$).

### Resolution Protocol

1. **Bước 1**: Đảm bảo đã đưa dữ liệu qua `StandardScaler()`.

2. **Bước 2**: Tăng tham số lặp: `LogisticRegression(max_iter=1000)`.

3. **Bước 3**: Nếu dữ liệu rất lớn hoặc thưa, đổi sang `solver='saga'`.

---

## 2. Preventing Data Leakage (Rò rỉ dữ liệu)

### Pitfall

Thực hiện các thao tác tiền xử lý dữ liệu (như `fit_transform()` của `StandardScaler` hoặc `SimpleImputer`) trên **toàn bộ dữ liệu** trước khi phân chia `train_test_split`.

Con đường rò rỉ: Trung bình $\mu$ và độ lệch chuẩn $\sigma$ của tập kiểm thử (Test Set) bị tính gộp vào tập huấn luyện, khiến điểm đánh giá mô hình cao bất thường nhưng thất bại khi triển khai thực tế.

### Strict Solution

Thực hiện `fit` duy nhất trên tập Train, sau đó áp dụng `transform` cho tập Test. Cách tốt nhất là gói toàn bộ luồng trong `sklearn.pipeline.Pipeline`:

```python
# QUY TRÌNH CHUẨN KHÔNG RÒ RỈ DỮ LIỆU
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Đóng gói mô hình vào Pipeline
secure_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Cross-Validation sẽ tự động mã hóa độc lập từng fold, tuyệt đối không bị leak
scores = cross_val_score(
    secure_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='f1'
)
```

### Minh Họa Bằng Số Liệu Cụ Thể (Tự Tính Tay, Không Dùng `StandardScaler`)

Để thấy rõ "rò rỉ" nằm ở đâu, ta tự tính tay $\mu$ và $\sigma$ theo 2 cách — **sai** (gộp cả Test) và **đúng** (chỉ dùng Train) — rồi so sánh:

```python
import numpy as np

# Giả lập một cột dữ liệu duy nhất để minh họa
np.random.seed(42)
X_train = np.random.normal(loc=50, scale=10, size=80)   # 80 mẫu Train
X_test = np.random.normal(loc=70, scale=15, size=20)     # 20 mẫu Test (phân phối khác Train)

# ---- CÁCH SAI: tính mu, sigma trên TOÀN BỘ dữ liệu (Train + Test) ----
X_all = np.concatenate([X_train, X_test])
mu_leak = X_all.mean()
sigma_leak = X_all.std()
X_train_leak = (X_train - mu_leak) / sigma_leak

# ---- CÁCH ĐÚNG: tính mu, sigma CHỈ trên Train ----
mu_correct = X_train.mean()
sigma_correct = X_train.std()
X_train_correct = (X_train - mu_correct) / sigma_correct

print(f"mu tính đúng (chỉ Train)  : {mu_correct:.3f}")
print(f"mu bị rò rỉ (Train+Test)  : {mu_leak:.3f}")
print(f"Sai lệch do leakage        : {abs(mu_correct - mu_leak):.3f}")
```

**Quan sát**: vì `X_test` có phân phối khác (`loc=70` thay vì `50`), `mu_leak` bị "kéo lệch" về phía thông tin của tập Test — đây chính là cách mô hình "nhìn trộm" được đặc điểm của dữ liệu nó chưa từng thấy, khiến điểm đánh giá trên Test trông cao giả tạo.

---

## 3. Multicollinearity Detection with VIF

### Concept

Đa cộng tuyến không làm giảm khả năng dự đoán toàn cục của mô hình, nhưng nó làm biến động cực đại các hệ số trọng số $\mathbf{w}$, khiến ta không thể giải thích được mức độ quan trọng của từng đặc trưng.

### Variance Inflation Factor (VIF)

Chỉ số VIF đo lường mức độ gia tăng phương sai của hệ số ước lượng do đa cộng tuyến:

$$
\text{VIF}_i = \frac{1}{1 - R_i^2}
$$

Trong đó $R_i^2$ là hệ số xác định khi hồi quy đặc trưng $x_i$ theo tất cả các đặc trưng còn lại.

- $\text{VIF} = 1$: Không có tương quan.
- $1 < \text{VIF} < 5$: Tương quan vừa phải (An toàn).
- $\text{VIF} > 10$: Đa cộng tuyến nghiêm trọng $\rightarrow$ Cần loại bỏ thuộc tính hoặc áp dụng L2 Regularization.

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(X_df):
    """
    Tính toán chỉ số VIF cho dataframe đầu vào
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(X_df.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)
```

### Tự Cài Đặt VIF (Không Dùng `statsmodels`)

Bản trên gọi thẳng `variance_inflation_factor` của `statsmodels`. Để hiểu rõ $R_i^2$ trong công thức VIF đến từ đâu, ta tự cài đặt hồi quy tuyến tính bằng **Phương trình chuẩn (Normal Equation)** — $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ — để tự tính $R_i^2$ rồi suy ra VIF:

```python
import numpy as np
import pandas as pd

def r_squared_scratch(X_others, y_feature):
    """
    Tự hồi quy tuyến tính (Normal Equation) của đặc trưng x_i theo các đặc trưng
    còn lại, sau đó tính R^2, không dùng statsmodels hay sklearn.
    """
    n = X_others.shape[0]
    X_with_bias = np.column_stack([np.ones(n), X_others])  # thêm cột bias

    # Phương trình chuẩn: beta = (X^T X)^-1 X^T y
    XtX = X_with_bias.T @ X_with_bias
    XtX_inv = np.linalg.pinv(XtX)  # dùng pinv để tránh lỗi ma trận suy biến
    beta = XtX_inv @ X_with_bias.T @ y_feature

    y_pred = X_with_bias @ beta
    ss_res = np.sum((y_feature - y_pred) ** 2)          # Tổng bình phương phần dư
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
        y_feature = X_values[:, i]                          # đặc trưng x_i làm "target"
        X_others = np.delete(X_values, i, axis=1)            # các đặc trưng còn lại

        r2_i = r_squared_scratch(X_others, y_feature)
        vif_i = 1 / (1 - r2_i + 1e-15)
        results.append({"Feature": col, "VIF": vif_i})

    return pd.DataFrame(results).sort_values(by="VIF", ascending=False)


# --- Đối chiếu với statsmodels ---
vif_scratch = check_multicollinearity_scratch(X_df)
vif_statsmodels = check_multicollinearity(X_df)  # hàm dùng statsmodels ở trên
print("So sánh 2 kết quả (giá trị VIF phải xấp xỉ bằng nhau):")
print(vif_scratch.merge(vif_statsmodels, on="Feature", suffixes=("_scratch", "_statsmodels")))
```

**Lưu ý**: bản tự cài đặt dùng `np.linalg.pinv` (pseudo-inverse) thay vì nghịch đảo ma trận thông thường để tránh lỗi khi các đặc trưng cộng tuyến gần như hoàn hảo (ma trận $\mathbf{X}^T\mathbf{X}$ gần suy biến) — đây cũng chính là biểu hiện toán học của hiện tượng đa cộng tuyến mà VIF đang muốn đo lường.

---

## 4. Production Readiness Checklist

Trước khi lưu file mô hình (.pkl hoặc .joblib) để đưa lên Server FastAPI/Streamlit, bạn cần xác nhận đầy đủ các mục sau:

- [ ] Tất cả các thuộc tính số đều đã được chuẩn hóa qua `StandardScaler`.
- [ ] Tất cả các biến phân loại đã qua `OneHotEncoder(drop='first')` để tránh Dummy Trap.
- [ ] Tham số `max_iter` đặt tối thiểu là $1000$.
- [ ] Tham số `random_state` được cố định một số nguyên cụ thể.
- [ ] Đã kiểm tra chỉ số VIF và không còn thuộc tính có $\text{VIF} > 10$.
- [ ] Luồng tiền xử lý và mô hình được đóng gói nguyên khối trong một `sklearn.pipeline.Pipeline`.
- [ ] Đã đánh giá hiệu năng mô hình trên tập Test hoàn toàn độc lập (Unseen Test Set).

---

## 5. Official References

Scikit-Learn Utilities & Pitfalls Guide:

https://scikit-learn.org/stable/common_pitfalls.html
