---
topic: "Logistic Regression"
subtopic: "Common Pitfalls, Troubleshooting & Best Practices"
level: "Intermediate/Advanced"
doc_id: "logreg_05"
sources:
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
