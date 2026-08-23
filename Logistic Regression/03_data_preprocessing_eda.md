---
topic: "Logistic Regression"
subtopic: "Data Preprocessing, Feature Engineering & Pipeline Integration"
level: "Intermediate"
doc_id: "logreg_03"
source_url: "https://scikit-learn.org/stable/modules/preprocessing.html"
scikit_learn_version: "1.4"
key_concepts:
  - "Chuẩn hóa đặc trưng (StandardScaler & RobustScaler)"
  - "Bẫy biến giả & Mã hóa biến phân loại"
  - "Đa cộng tuyến & Điền khuyết dữ liệu"
  - "Tự cài đặt Chuẩn hóa & One-Hot Encoding từ đầu"
  - "Xử lý mất cân bằng lớp (Trọng số Loss & Ngưỡng quyết định)"
  - "Pipeline thực chiến với ColumnTransformer"
---

# Tiền Xử Lý Dữ Liệu & Pipeline Hoàn Chỉnh Cho Logistic Regression

---

## 1. Tại Sao Tiền Xử Lý Dữ Liệu Là BẮT BUỘC Với Logistic Regression?

Khác với các thuật toán dạng cây (Decision Tree, Random Forest) không quan tâm đến thang đo, Logistic Regression hoạt động dựa trên **Giải thuật Tối ưu Gradient** và tính **Tuyến tính**. Do đó, dữ liệu thô chưa xử lý sẽ trực tiếp phá hỏng mô hình:

- **Chênh lệch thang đo (Scale Differences):** Nếu cột `Tuổi` nằm trong khoảng $[18, 80]$ còn cột `Thu nhập` nằm trong khoảng $[5.000.000, 100.000.000]$, Gradient của `Thu nhập` sẽ hoàn toàn áp đảo `Tuổi`. Điều này khiến đường đi của Gradient Descent bị dao động hình zích-zắc, hội tụ cực kỳ chậm và làm cho phạt điều hòa (L1/L2 Regularization) bị thiên vị lệch lạc.
- **Đa cộng tuyến (Multicollinearity):** Khi 2 đặc trưng tương quan quá mạnh với nhau (ví dụ: `Chiều cao (cm)` và `Chiều cao (inch)`), ma trận tính toán bị nhiễu nặng khiến trọng số $\mathbf{w}$ bị bùng nổ phương sai (mô hình không ổn định, dữ liệu thay đổi nhỏ cũng làm trọng số đảo chiều).
- **Giá trị khuyết & Ngoại lai (Missing Values & Outliers):** Hàm Log-Loss phạt rất nặng (tiến ra vô cùng) cho các dự đoán sai. Chỉ cần một vài giá trị ngoại lai quá dị biệt cũng có thể kéo lệch toàn bộ đường ranh giới quyết định (Decision Boundary).

---

## 2. Quy Trình Tiền Xử Lý Chuẩn 3 Bước (Step-by-Step Protocol)

### Bước 1: Phát hiện và xử lý giá trị vô lý (Invalid & Missing Values)

Trong thực tế, dữ liệu khuyết không chỉ hiển thị dạng `NaN` hay `Null` mà thường bị lấp liếm bằng số `0`. Ví dụ trong dữ liệu y tế, các chỉ số như `Huyết áp` (RestingBP) hay `Cholesterol` bằng $0$ là bất hợp lý về mặt sinh học.

```python
import pandas as pd
import numpy as np

# Khởi tạo dữ liệu giả lập chứa lỗi
df = pd.DataFrame({
    'Age': [40, 49, 37, 54],
    'RestingBP': [140, 0, 130, 110],    # Số 0 ở đây là dữ liệu bị lỗi/khuyết!
    'Cholesterol': [289, 180, 0, 250]   # Số 0 ở đây là dữ liệu bị lỗi/khuyết!
})

# Chuyển các giá trị 0 phi lý thành NaN để tiến hành điền khuyết (Imputation)
invalid_cols = ['RestingBP', 'Cholesterol']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)
```

### Bước 2: Mã hóa biến phân loại (Categorical Encoding)

- **One-Hot Encoding:** Áp dụng cho biến định danh không có thứ tự (như `Giới tính`, `Thành phố`). Bắt buộc dùng tham số `drop='first'` để bỏ đi 1 cột phụ thuộc tuyến tính, giúp loại bỏ **Bẫy biến giả (Dummy Variable Trap)**.
- **Ordinal Encoding:** Áp dụng cho các biến có thứ tự rõ ràng (như `Trình độ học vấn`: *Cấp 3 < Đại học < Thạc sĩ*).

> **Giải thích "Bẫy biến giả":** Nếu cột `Giới tính` tạo ra 2 cột mới là `Is_Male` và `Is_Female`, ta luôn có `Is_Male + Is_Female = 1`. Việc biết `Is_Male` đã hoàn toàn đoán được `Is_Female`, gây ra hiện tượng đa cộng tuyến hoàn hảo. Vì vậy phải bỏ bớt 1 cột!

### Bước 3: Chuẩn hóa thang đo (Feature Scaling)

Ta dùng `StandardScaler` để biến đổi các thuộc tính về phân phối chuẩn với Trung bình $\mu = 0$ và Độ lệch chuẩn $\sigma = 1$:

$$
x' = \frac{x - \mu}{\sigma}
$$

- **Dữ liệu chuẩn:** Dùng `StandardScaler`.
- **Dữ liệu chứa nhiều nhiễu/Outliers:** Dùng `RobustScaler` (sử dụng Median và khoảng tứ phân vị IQR) để tránh bị giá trị ngoại lai kéo lệch thang đo.

---

## 3. Cài Đặt Thủ Công Bằng NumPy & Pandas (Scratch Implementation)

Để hiểu rõ bản chất toán học phía sau các hàm của Scikit-Learn, ta tự cài đặt lại các bước tiền xử lý này.

### 3.1. Tự cài đặt Standardization (thay thế `StandardScaler`)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standard_scaler_scratch(X_train, X_test):
    """
    Tự tính toán công thức: x' = (x - mu) / sigma

    LƯU Ý CỰC KỲ QUAN TRỌNG:
    mu và sigma CHỈ ĐƯỢC TÍNH TỪ X_train để tránh Rò rỉ dữ liệu (Data Leakage).
    Sau đó dùng chính mu và sigma đó để transform cho X_test.
    """
    # 1. Tính mu và sigma trên tập Train
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    # 2. Xử lý trường hợp sigma = 0 (cột chứa toàn giá trị hằng số) để tránh chia cho 0
    sigma_safe = np.where(sigma == 0, 1e-15, sigma)

    # 3. Áp dụng chuẩn hóa cho cả 2 tập
    X_train_scaled = (X_train - mu) / sigma_safe
    X_test_scaled = (X_test - mu) / sigma_safe  # Dùng lại mu và sigma của Train!

    return X_train_scaled, X_test_scaled

# --- KIỂM THỬ VỚI SCIKIT-LEARN ---
X_tr = np.array([[18, 5000], [45, 12000], [60, 25000]], dtype=float)
X_te = np.array([[30, 8000]], dtype=float)

# Bản Scratch
X_tr_scratch, X_te_scratch = standard_scaler_scratch(X_tr, X_te)

# Bản Sklearn
scaler = StandardScaler()
X_tr_sklearn = scaler.fit_transform(X_tr)
X_te_sklearn = scaler.transform(X_te)

# So sánh độ lệch
print("Sai lệch tối đa tập Train:", np.max(np.abs(X_tr_scratch - X_tr_sklearn)))
print("Sai lệch tối đa tập Test :", np.max(np.abs(X_te_scratch - X_te_sklearn)))
```

### 3.2. Tự cài đặt One-Hot Encoding (thay thế `OneHotEncoder(drop='first')`)

```python
def one_hot_encode_scratch(df, categorical_cols, drop_first=True):
    """
    Tự động chuyển đổi các cột dạng phân loại thành dạng 0/1.
    Loại bỏ cột đầu tiên (drop_first=True) để tránh Dummy Variable Trap.
    """
    df_encoded = df.copy()

    for col in categorical_cols:
        # Lấy danh sách các giá trị phân biệt đã sắp xếp
        categories = sorted(df_encoded[col].dropna().unique())

        # Nếu drop_first=True, bỏ qua nhóm đầu tiên
        categories_to_encode = categories[1:] if drop_first else categories

        for cat in categories_to_encode:
            new_col_name = f"{col}_{cat}"
            # Tạo cột mới chứa giá trị 1 nếu trùng khớp, ngược lại là 0
            df_encoded[new_col_name] = (df_encoded[col] == cat).astype(int)

        # Xóa cột phân loại gốc
        df_encoded = df_encoded.drop(columns=[col])

    return df_encoded

# --- VÍ DỤ MINH HỌA ---
df_demo = pd.DataFrame({'Sex': ['M', 'F', 'F', 'M'], 'Age': [40, 49, 37, 54]})
df_encoded = one_hot_encode_scratch(df_demo, categorical_cols=['Sex'], drop_first=True)
print("\nKết quả One-Hot Encoding tự viết:")
print(df_encoded)
```

| Trường hợp | Ưu điểm bản Scratch | Hạn chế bản Scratch (Vì sao thực tế dùng Sklearn?) |
| --- | --- | --- |
| **Standardization** | Dễ hiểu, nắm trọn công thức toán gốc. | Chưa tối ưu cho ma trận thưa (Sparse Matrix), không hỗ trợ ghi nhớ tham số vào object tiện lợi. |
| **One-Hot Encoding** | Thao tác trực tiếp trên Pandas DataFrame rất trực quan. | Dễ lỗi nếu tập `Test` xuất hiện nhãn mới mà tập `Train` chưa từng thấy (`handle_unknown='ignore'`). |

---

## 4. Xử Lý Mất Cân Bằng Nhãn (Class Imbalance Mitigation)

Khi tập dữ liệu bị lệch nặng (Ví dụ: 95% mẫu Không bệnh - Lớp 0, 5% mẫu Có bệnh - Lớp 1), mô hình có xu hướng dự đoán toàn bộ là Lớp 0 để đạt Accuracy 95%. Hai cách xử lý hiệu quả nhất bao gồm:

### 1. Điều chỉnh trọng số Loss Function (`class_weight='balanced'`)

Trọng số $w_k$ của lớp $k$ sẽ được tự động phạt nặng hơn nếu số lượng mẫu của lớp đó quá ít:

$$
w_k = \frac{N}{K \cdot N_k}
$$

- $N$: Tổng số mẫu dữ liệu.
- $K$: Tổng số lớp (phân loại nhị phân thì $K = 2$).
- $N_k$: Số lượng mẫu thuộc lớp $k$.

> **Ý nghĩa:** Nếu lớp 1 chỉ có 5 mẫu trong tổng số 100 mẫu, trọng số phạt sai sót của lớp 1 sẽ cao gấp 10 lần so với lớp 0!

### 2. Tối ưu ngưỡng quyết định (Decision Threshold Tuning)

Mặc định, Logistic Regression cắt ngưỡng tại $0.5$ ($P \ge 0.5 \rightarrow$ Lớp 1). Trong bài toán Y tế hoặc Phát hiện gian lận, ta chủ động **hạ thấp ngưỡng** (ví dụ xuống $0.2$ hoặc $0.3$) để bắt tối đa các ca bệnh, chấp nhận đánh đổi một ít Precision để lấy **Recall cao**:

```python
# Giả sử y_proba là xác suất dự đoán lớp 1 từ model.predict_proba(X)[:, 1]
custom_threshold = 0.3
y_pred_custom = (y_proba >= custom_threshold).astype(int)
```

---

## 5. Quy Trình Thực Chiến Hoàn Chỉnh VỚI `ColumnTransformer` & `Pipeline`

Trong dự án thực tế, ta sử dụng `ColumnTransformer` để áp dụng các bước xử lý riêng biệt cho cột Số và cột Chữ, sau đó đóng gói tất cả vào một `Pipeline` duy nhất.

```python
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Tạo dữ liệu mô phỏng dạng DataFrame chứa cả cột Số và Cột Chữ
np.random.seed(42)
n_samples = 1000

df_data = pd.DataFrame({
    'Age': np.random.randint(20, 70, size=n_samples),
    'Cholesterol': np.random.choice([0, 180, 220, 280, 310], size=n_samples),  # Có số 0 bị lỗi
    'Sex': np.random.choice(['M', 'F'], size=n_samples),
    'ChestPain': np.random.choice(['ATA', 'NAP', 'ASY'], size=n_samples),
    'Target': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # Mất cân bằng nhãn (80/20)
})

# Chuyển số 0 ở Cholesterol thành NaN
df_data['Cholesterol'] = df_data['Cholesterol'].replace(0, np.nan)

X = df_data.drop(columns=['Target'])
y = df_data['Target']

# 2. Phân chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Phân loại các cột đặc trưng
num_cols = ['Age', 'Cholesterol']
cat_cols = ['Sex', 'ChestPain']

# 4. Xây dựng Pipeline riêng cho cột Số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Điền khuyết bằng Median
    ('scaler', StandardScaler())                     # Chuẩn hóa thang đo
])

# 5. Xây dựng Pipeline riêng cho cột Chữ
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Điền khuyết bằng giá trị xuất hiện nhiều nhất
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Mã hóa One-Hot, bỏ cột đầu
])

# 6. Ghép 2 nhánh tiền xử lý bằng ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# 7. Đóng gói toàn bộ Preprocessor + Model vào Full Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced',  # Xử lý mất cân bằng nhãn
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ))
])

# 8. Huấn luyện toàn bộ Pipeline (Chỉ gọi .fit trên X_train!)
full_pipeline.fit(X_train, y_train)

# 9. Dự đoán trên tập Test
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (PRODUCTION PIPELINE) ===")
print(classification_report(y_test, y_pred))
print(f"Chỉ số ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

## 6. Khung Dẫn Dắt Bài Tập & Thực Hành (Exercise Framework)

Dưới đây là các dạng bài tập tự luyện giúp bạn làm chủ phần tiền xử lý dữ liệu:

### Dạng 1: Bài tập Tự cài đặt Min-Max Scaler (Scratch)

- **Yêu cầu:** Hãy tự viết hàm `min_max_scaler_scratch(X_train, X_test)` thực hiện công thức:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

- **Gợi ý:** Tính $x_{\min}$ và $x_{\max}$ trên tập `X_train`, sau đó áp dụng biến đổi cho cả `X_train` và `X_test`.

### Dạng 2: Bài tập Phát hiện Rò rỉ Dữ liệu (Data Leakage Hunt)

- **Yêu cầu:** Hãy chỉ ra lỗi sai nguy hiểm trong đoạn code sau và sửa lại cho đúng bằng `Pipeline`:

```python
# ĐOẠN CODE LỖI:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa TOÀN BỘ dữ liệu trước khi split!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

### Dạng 3: Bài tập Tối ưu Ngưỡng Quyết Định (Threshold Tuning)

- **Yêu cầu:** Với Pipeline đã huấn luyện ở Mục 5, hãy thử nghiệm các ngưỡng $Threshold \in [0.1, 0.2, 0.3, 0.4, 0.5]$ và vẽ đồ thị sự thay đổi giữa hai chỉ số **Recall** và **Precision**.

---

## 7. Tài Liệu Tham Khảo (Official References)

- Scikit-Learn User Guide: [Preprocessing Data Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
- Scikit-Learn API Reference: [sklearn.compose.ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly Media.
