---
topic: "Decision Tree"
subtopic: "Scikit-Learn API, Hyperparameters & Tuning"
level: "Intermediate"
doc_id: "dt_02"
sources:
  - "Scikit-Learn v1.4 API Reference: sklearn.tree.DecisionTreeClassifier"
key_concepts:
  - "DecisionTreeClassifier"
  - "max_depth"
  - "min_samples_split"
  - "min_samples_leaf"
  - "class_weight"
---

# Scikit-Learn API & Hyperparameter Matrix

## 1. Main Class Specification
Lớp chính: `sklearn.tree.DecisionTreeClassifier`.
Các phương thức cốt lõi tương tự như Logistic Regression:
* `.fit(X, y)`: Xây dựng cây từ tập huấn luyện.
* `.predict(X)`: Suy luận nhãn bằng cách duyệt từ rễ đến lá.
* `.predict_proba(X)`: Trả về xác suất thuộc các lớp (chính là tỉ lệ $p_{mk}$ của các lớp tại nút lá chứa mẫu đó).
* `.apply(X)`: Trả về chỉ số (index) của nút lá mà mỗi mẫu dữ liệu rơi vào.

## 2. Hyperparameter Configuration Matrix
Mô hình cây tự do (không tinh chỉnh) mặc định sẽ học thuộc lòng (memorize) toàn bộ dữ liệu, dẫn đến tỷ lệ lỗi trên tập Test rất cao. Dưới đây là các tham số dùng để kiểm soát (Regularization):

| Tham số | Kiểu dữ liệu | Mặc định | Mức độ ảnh hưởng & Hướng dẫn sử dụng |
| :--- | :--- | :--- | :--- |
| `criterion` | `str` | `'gini'` | Hàm đánh giá phân tách: `'gini'`, `'entropy'`, `'log_loss'`. Thường `'gini'` đủ tốt và tính toán nhanh hơn. |
| `max_depth` | `int` | `None` | Rất quan trọng. Độ sâu tối đa của cây. Nếu `None`, cây phát triển cho đến khi các lá có Gini=0. Nên bắt đầu GridSearch từ $[3, 5, 7, 10]$. |
| `min_samples_split` | `int` / `float` | `2` | Số mẫu tối thiểu một nút cần có để được phép tách tiếp. Tăng lên (ví dụ $10-50$) giúp hạn chế học nhiễu. |
| `min_samples_leaf` | `int` / `float` | `1` | Số lượng mẫu tối thiểu phải tồn tại ở một nút lá. Tăng lên (ví dụ $5-20$) giúp làm mượt dự đoán, triệt tiêu các lá lẻ tẻ. |
| `max_features` | `int` / `str` | `None` | Số lượng features tối đa xét đến khi tách. Dùng `'sqrt'` hoặc `'log2'` để giảm tính toán và tránh cây bị phụ thuộc quá nhiều vào 1 feature. |
| `class_weight`| `dict` / `str`| `None` | Xử lý mất cân bằng lớp. Cấu hình `'balanced'` tự động nhân trọng số tỷ lệ nghịch với tần suất lớp vào công thức tính Gini. |

## 3. Complete Production Executable Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Dữ liệu giả lập
X, y = make_classification(n_samples=1500, n_features=12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Định nghĩa Pipeline
# Lưu ý: Bỏ qua StandardScaler vì Decision Tree không cần
pipeline = Pipeline([
    ('model', DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ))
])

# 3. Huấn luyện và Đánh giá
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
