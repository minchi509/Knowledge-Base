---
topic: "Logistic Regression"
subtopic: "Data Preprocessing, Feature Engineering & Pipeline Integration"
level: "Intermediate"
doc_id: "logreg_03"
sources:
  - "Scikit-Learn User Guide v1.4: Preprocessing Data"
---

# Data Preprocessing & EDA Guidelines for Logistic Regression

## 1. Why Preprocessing is Mandatory for Logistic Regression

Logistic Regression dựa vào các giải thuật tối ưu độ dốc (Gradient-based Optimizers) và giả định sự độc lập/tuyến tính giữa các thuộc tính:

* **Chênh lệch Thang đo (Scale Differences)**: Nếu biến $A$ có miền giá trị $[0, 1]$ và biến $B$ có miền giá trị $[100, 100000]$, gradient của biến $B$ sẽ áp đảo biến $A$. Điều này khiến quá trình tìm nghiệm lằn rằn, chậm hội tụ và làm cho phạt điều hòa (L1/L2 Penalty) bị thiên vị.

* **Đa cộng tuyến (Multicollinearity)**: Các thuộc tính tương quan cao làm tăng phương sai của các hệ số trọng số $\mathbf{w}$, dẫn đến mô hình không ổn định.

* **Giá trị khuyết & Ngoại lai (Missing Values & Outliers)**: Log-Loss rất nhạy cảm với các mẫu bị nhiễu xa trung vị.

---

## 2. Step-by-Step Preprocessing Protocol

### Step 1: Handling Invalid and Missing Values

Trong các bộ dữ liệu y tế (ví dụ: `Heart Failure Prediction`), các chỉ số sinh lý như `RestingBP` (huyết áp) hoặc `Cholesterol` mang giá trị $0$ là bất hợp lý về mặt sinh học. Những giá trị này phải được gán lại thành `np.nan` và tiến hành Imputation bằng Median.

```python
import pandas as pd
import numpy as np

# Giả lập làm sạch dữ liệu bất thường
df = pd.DataFrame({
    'Age': [40, 49, 37, 54],
    'RestingBP': [140, 0, 130, 110],       # 0 là giá trị bất thường
    'Cholesterol': [289, 180, 0, 250]      # 0 là giá trị bất thường
})

# Thay thế giá trị 0 thành NaN ở các cột sinh lý
invalid_cols = ['RestingBP', 'Cholesterol']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)
```

### Step 2: Categorical Encoding

**One-Hot Encoding:** Áp dụng cho các biến phân loại không có thứ tự (Nominal Variables). Bắt buộc bật `drop='first'` để loại bỏ cột phụ thuộc tuyến tính, tránh hiện tượng Dummy Variable Trap.

**Ordinal Encoding:** Áp dụng cho các biến có thứ tự rõ ràng.

### Step 3: Feature Scaling

Sử dụng `StandardScaler` để đưa các thuộc tính liên tục về phân phối chuẩn có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$:

$$
x' = \frac{x - \mu}{\sigma}
$$

Đối với dữ liệu chứa nhiều Outliers, sử dụng `RobustScaler` (dựa trên Median và Interquartile Range - IQR) để tránh làm lệch giá trị chuẩn hóa.

---

## 3. Class Imbalance Mitigation

Khi số lượng mẫu thuộc lớp $0$ và lớp $1$ chênh lệch lớn (ví dụ: 90% lành tính, 10% mắc bệnh):

### Loss Function Weighting (`class_weight='balanced'`)

Scikit-learn sẽ tự động gán trọng số $w_k$ cho mỗi lớp theo công thức:

$$
w_k = \frac{N}{K \cdot N_k}
$$

Trong đó $N$ là tổng số mẫu, $K$ là số lớp, và $N_k$ là số mẫu của lớp $k$.

### Decision Threshold Tuning

Dựa vào bài toán thực tế, thay vì cố định ngưỡng $0.5$, ta hạ thấp ngưỡng quyết định (ví dụ: xuống $0.3$) để ưu tiên tăng độ triệu hồi (Recall) cho lớp bệnh lý.

---

## 4. Production Ready Preprocessing Script (ColumnTransformer)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Danh sách các cột theo thuộc tính
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
cat_cols = ['Sex', 'ChestPainType', 'ExerciseAngina']

# 2. Pipeline xử lý cột số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3. Pipeline xử lý cột phân loại
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# 4. Tích hợp ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
```

---

## 5. Official References

Scikit-Learn User Guide: Preprocessing Data

https://scikit-learn.org/stable/modules/preprocessing.html
