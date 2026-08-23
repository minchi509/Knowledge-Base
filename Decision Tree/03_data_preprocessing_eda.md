---
topic: "Decision Tree"
subtopic: "Data Preprocessing, Encoding & Feature Importance"
level: "Intermediate"
doc_id: "dt_03"
source_url: "https://scikit-learn.org/stable/modules/tree.html"
key_concepts:
  - "Tính bất biến với thang đo (Scale Invariance)"
  - "Thiên vị do biến có nhiều giá trị (High Cardinality Bias)"
  - "So sánh Ordinal Encoding và One-Hot Encoding"
  - "Độ quan trọng Gini (Gini Importance / MDI)"
  - "Độ quan trọng theo hoán vị (Permutation Importance)"
---

# Hướng Dẫn Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng

## 1. Tính Bất Biến Với Thang Đo (Scale Invariance)

Một trong những ưu điểm lớn nhất của Cây quyết định (Decision Tree) so với các thuật toán học máy khác là **hoàn toàn không yêu cầu chuẩn hóa dữ liệu** (Không cần `StandardScaler` hay `MinMaxScaler`).

### Tại sao Cây quyết định không quan tâm đến Thang đo?

- **Cơ chế phân tách đơn biến (Axis-aligned split):** Tại mỗi nút, cây chỉ xem xét **một đặc trưng đơn lẻ** $x_j$ và so sánh nó với một ngưỡng $t$ ($x_j \le t$).
- **Biến đổi đơn điệu (Monotonic Transformation):** Việc nhân một cột với $1000$ (chuyển từ Km sang m) hoặc lấy hàm $\log(x)$ không làm thay đổi thứ tự tương đối giữa các điểm dữ liệu. Do đó, điểm cắt ngưỡng $t$ tối ưu vẫn giữ nguyên vị trí phân loại của nó.

### So sánh yêu cầu Tiền xử lý dữ liệu:

| Thuật toán | Cần Chuẩn hóa Thang đo? | Lý do |
| --- | --- | --- |
| **Decision Tree / Random Forest** | **KHÔNG** | Phân tách dựa trên thứ tự xếp hạng (Rank-based) của từng cột độc lập. |
| **Logistic Regression / Neural Net** | **CÓ** | Dùng Gradient Descent; nếu thang đo lệch nhau, mặt mất mát sẽ bị méo. |
| **KNN / SVM / K-Means** | **CÓ** | Dựa trên khoảng cách không gian (Euclidean); cột có giá trị lớn sẽ áp đảo. |

---

## 2. Mã Hóa Biến Phân Loại & Thiên Vị Do High Cardinality

Mặc dù bất biến với thang đo, `DecisionTreeClassifier` trong Scikit-Learn vẫn yêu cầu đầu vào phải là dạng số (`float` hoặc `int`). Cách bạn biến đổi các biến phân loại (Categorical Features) thành dạng số sẽ ảnh hưởng trực tiếp đến chất lượng của cây.

### 2.1. One-Hot Encoding vs Ordinal Encoding

- **One-Hot Encoding (`OneHotEncoder`):** Tách 1 cột phân loại thành $N$ cột nhị phân ($0$ và $1$).
  - *Nhược điểm với Cây:* Tạo ra ma trận dữ liệu thưa thớt (sparse matrix). Cây phải duyệt qua rất nhiều cột phụ, làm tăng độ sâu của cây và làm suy giảm khả năng chọn đặc trưng gốc.
- **Ordinal Encoding (`OrdinalEncoder`):** Gán mỗi giá trị phân loại thành một số nguyên ($0, 1, 2, 3...$).
  - *Ưu điểm với Cây:* Giữ nguyên số lượng cột, cây vẫn có thể thực hiện phép so sánh $\le$ để chia nhóm các nhãn hiệu quả mà không làm "nổ" kích thước cây.

### 2.2. Hiện tượng Thiên vị Kích thước Tập giá trị (High Cardinality Bias)

**High Cardinality** là thuật ngữ chỉ các cột có quá nhiều giá trị duy nhất (ví dụ: `Mã_Bưu_Điện`, `ID_Nguoi_Dung`, `Ngay_Sinh`).

- **Nguyên nhân:** Một biến liên tục hoặc một biến phân loại có $100$ giá trị duy nhất sẽ cung cấp tới $99$ điểm cắt ngưỡng $t$ ứng viên cho thuật toán CART thử nghiệm. Trong khi đó, một biến nhị phân (Đúng/Sai) chỉ cung cấp đúng $1$ điểm cắt.
- **Hậu quả:** Thuật toán tham lam (Greedy Algorithm) sẽ **bị thiên vị**, luôn ưu tiên chọn các biến có High Cardinality để phân tách vì chúng dễ dàng tìm được một điểm cắt ngẫu nhiên làm giảm độ bất thuần (Impurity) trên tập Train. Điều này khiến mô hình đánh giá sai tầm quan trọng của đặc trưng và gây ra Overfitting nghiêm trọng.

---

## 3. Trích Xuất Độ Quan Trọng Đặc Trưng (Feature Importance)

Sau khi huấn luyện (`.fit()`), mô hình cung cấp thuộc tính `model.feature_importances_`. Đây chính là chỉ số **Gini Importance** (hay còn gọi là **Mean Decrease in Impurity - MDI**).

### 3.1. Cơ sở Toán học

Mức độ giảm độ bất thuần tại nút $m$ khi tách bằng đặc trưng $j$:

$$
\Delta H(m) = H(m) - \left( \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R) \right)
$$

Độ quan trọng chưa chuẩn hóa của đặc trưng $j$ là tổng giảm Impurity trên **tất cả các nút** mà đặc trưng $j$ được chọn để tách, có trọng số là số lượng mẫu $N_m$ tại nút đó:

$$
I(j) = \sum_{m \in \text{Nodes split on } j} N_m \cdot \Delta H(m)
$$

Độ quan trọng chuẩn hóa (được trả về bởi `feature_importances_`):

$$
\text{Importance}(j) = \frac{I(j)}{\sum_{k} I(k)} \quad \left(\text{Tổng tất cả các Feature Importances} = 1.0\right)
$$

### 3.2. Hạn chế của Gini Importance (MDI) & Giải pháp

1. **Bị lệch bởi High Cardinality:** Như đã giải thích ở Mục 2.2, biến có nhiều giá trị duy nhất sẽ tự động có Gini Importance cao hơn thực tế.
2. **Tính toán trên tập Train:** MDI chỉ đo đạc mức độ giảm độ bất thuần trên tập dữ liệu huấn luyện, không phản ánh khả năng dự đoán trên tập dữ liệu mới (Test Set).
3. **Giải pháp thay thế:** Sử dụng **Permutation Importance** (`sklearn.inspection.permutation_importance`). Phương pháp này đo lường sự sụt giảm độ chính xác của mô hình trên tập Validation/Test khi bị xáo trộn ngẫu nhiên giá trị của một cột đặc trưng.

---

## 4. Mã Nguồn Hoàn Chỉnh: Mã Hóa Dữ Liệu & Trực Quan Hóa Độ Quan Trọng Đặc Trưng

Đoạn mã hoàn chỉnh dưới đây minh họa quy trình: Mã hóa dữ liệu phân loại, huấn luyện cây, trích xuất và vẽ biểu đồ độ quan trọng đặc trưng bằng `matplotlib`.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# 1. Tạo tập dữ liệu giả lập có cả biến số và biến chữ (Categorical)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Thu_Nhap': np.random.normal(15, 5, n_samples),          # Continuous
    'Tuoi': np.random.randint(18, 65, size=n_samples),        # Continuous
    'Trinh_Do': np.random.choice(['Bieu_Thong', 'Dai_Hoc', 'Thac_Si'], size=n_samples),  # Categorical
    'Khu_Vuc': np.random.choice(['Mien_Bac', 'Mien_Trung', 'Mien_Nam'], size=n_samples),  # Categorical
})

# Nhãn mục tiêu y (Phân loại nhị phân)
y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

# 2. Xử lý Tiền xử lý (Preprocessing Pipeline)
categorical_cols = ['Trinh_Do', 'Khu_Vuc']
numerical_cols = ['Thu_Nhap', 'Tuoi']

# Dùng OrdinalEncoder cho các biến Categorical trong mô hình Cây
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), categorical_cols),
        ('num', 'passthrough', numerical_cols)  # Bỏ qua biến số, không cần scale
    ]
)

# Biến đổi X
X_processed = preprocessor.fit_transform(data)
feature_names = categorical_cols + numerical_cols

# Chia tập Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 3. Huấn luyện Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# 4. Trích xuất và Trực quan hóa Feature Importance
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sắp xếp giảm dần

    plt.figure(figsize=(9, 5))
    plt.title("Mức độ quan trọng của các Đặc trưng (Gini Importance - MDI)")
    plt.bar(range(len(importances)), importances[indices], align="center", color="skyblue", edgecolor="navy")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=15)
    plt.xlabel("Tên Đặc trưng")
    plt.ylabel("Chỉ số Importance (Tổng = 1.0)")

    # Hiển thị giá trị số trên đầu mỗi cột
    for i, idx in enumerate(indices):
        plt.text(i, importances[idx] + 0.01, f"{importances[idx]:.3f}", ha='center')

    plt.tight_layout()
    plt.show()

# Gọi hàm vẽ biểu đồ
plot_feature_importances(dt_model, feature_names)
```

---

## 5. Checklist Tiền Xử Lý Cho Mô Hình Cây (Quy Tắc Cốt Lõi)

- **Bỏ qua Scaling:** Không mất thời gian áp dụng `StandardScaler` hay `MinMaxScaler`.
- **Ưu tiên Ordinal Encoding:** Với Cây quyết định đơn lẻ hay Ensemble (Random Forest, XGBoost), nên ưu tiên `OrdinalEncoder` hơn là `OneHotEncoder` để tránh làm thưa thớt dữ liệu.
- **Cẩn trọng với High Cardinality:** Nếu một biến phân loại có quá nhiều giá trị (ví dụ >100 nhóm), cân nhắc dùng Target Encoding hoặc gom nhóm các nhãn hiếm lại trước khi cho vào mô hình.
- **Dùng Permutation Importance để kiểm tra lại:** Nếu nghi ngờ Gini Importance bị thiên vị, hãy tính `permutation_importance` trên tập Test để đánh giá chính xác giá trị thực sự của từng cột.
