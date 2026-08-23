---
topic: "Decision Tree"
subtopic: "Scikit-Learn API, Hyperparameters & Tuning"
level: "Intermediate"
doc_id: "dt_02"
source_url: "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
key_concepts:
  - "Lớp DecisionTreeClassifier"
  - "Tham số max_depth (Độ sâu tối đa)"
  - "Tham số min_samples_split (Số mẫu tối thiểu để tách nút)"
  - "Tham số min_samples_leaf (Số mẫu tối thiểu tại nút lá)"
  - "Xử lý mất cân bằng lớp (class_weight)"
  - "Tinh chỉnh siêu tham số với GridSearchCV"
---

# API Scikit-Learn & Ma Trận Siêu Tham Số

## 1. Đặc Tả Lớp (`sklearn.tree.DecisionTreeClassifier`)

Lớp chính được sử dụng để xây dựng mô hình Cây quyết định cho bài toán Phân loại trong Python là `DecisionTreeClassifier`.

### Các Phương Thức Cốt Lõi

- **`.fit(X, y)`**: Xây dựng cây quyết định từ tập dữ liệu huấn luyện.
- **`.predict(X)`**: Dự đoán nhãn lớp cho các mẫu mới bằng cách cho mẫu chạy từ nút gốc (Root) xuống nút lá (Leaf).
- **`.predict_proba(X)`**: Trả về xác suất thuộc từng lớp. Tỷ lệ này chính là phân phối xác suất $p_{mk}$ của các lớp nhãn tại nút lá mà mẫu đó rơi vào.
- **`.apply(X)`**: Trả về chỉ số (Index) của nút lá mà mỗi mẫu dữ liệu rơi vào. Rất hữu ích khi muốn phân tích cấu trúc cây.
- **`.cost_complexity_pruning_path(X, y)`**: Tính toán chuỗi tham số phạt độ phức tạp $\alpha$ (`ccp_alpha`) phục vụ cho kỹ thuật Cắt tỉa sau (Post-pruning).

---

## 2. Ma Trận Siêu Tham Số & Cách Chống Quá Khớp

Một cây quyết định mặc định không giới hạn sẽ tự do phát triển cho đến khi nhớ sạch toàn bộ tập train (Overfitting). Để kiểm soát mô hình, ta điều chỉnh các siêu tham số (Hyperparameters) theo các nhóm chức năng sau:

| Siêu tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa trực quan & Hướng dẫn sử dụng |
| --- | --- | --- | --- |
| `criterion` | `str` | `'gini'` | Tiêu chí đo độ bất thuần: `'gini'`, `'entropy'`, hoặc `'log_loss'`. Thường `'gini'` chạy nhanh hơn và cho kết quả tương đương. |
| `max_depth` | `int` | `None` | **Độ sâu tối đa của cây.** Giới hạn số lượng câu hỏi nối tiếp. Bắt đầu thử từ khoảng $[3, 5, 7, 10]$ để tránh cây quá sâu. |
| `min_samples_split` | `int` / `float` | `2` | **Số mẫu tối thiểu tại một nút nội bộ để ĐƯỢC PHÉP TÁCH.** Tăng lên (ví dụ $10 - 50$) để chặn cây tạo nhánh lẻ tẻ. |
| `min_samples_leaf` | `int` / `float` | `1` | **Số mẫu tối thiểu BẮT BUỘC có tại một nút lá.** Tăng lên (ví dụ $5 - 20$) giúp làm mượt dự đoán, triệt tiêu lá chứa mẫu nhiễu. |
| `max_features` | `int` / `str` | `None` | Số lượng đặc trưng tối đa được xét khi tìm phép tách. Dùng `'sqrt'` hoặc `'log2'` để giảm tính toán và tăng tính đa dạng. |
| `class_weight` | `dict` / `str` | `None` | **Xử lý mất cân bằng lớp (Imbalanced Data).** Truyền `'balanced'` để tự động tăng trọng số cho các lớp yếu/ít mẫu. |

### Phân Biệt `min_samples_split` Và `min_samples_leaf` (Dành Cho Người Mới)

- **`min_samples_split` (Điều kiện Nút Cha):** Kiểm tra xem nút hiện tại có đủ dữ liệu để *tiến hành chia câu hỏi* hay không.
- **`min_samples_leaf` (Điều kiện Nút Con):** Đảm bảo rằng *sau khi chia*, cả 2 nút con sinh ra đều phải chứa tối thiểu số lượng mẫu quy định. Nếu 1 trong 2 nút con không đủ, phép chia sẽ bị hủy.

---

## 3. Tinh Chỉnh Siêu Tham Số Với `GridSearchCV`

Thay vì đoán mò siêu tham số, ta sử dụng **GridSearchCV** để thử nghiệm tự động tất cả các kết hợp và tìm ra bộ tham số tối ưu nhất dựa trên Kiểm chứng chéo (Cross-Validation).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Tải dữ liệu ví dụ (Ung thư vú)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Khai báo không gian siêu tham số cần thử nghiệm (Param Grid)
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# 3. Khởi tạo GridSearchCV với 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# 4. Huấn luyện để tìm bộ tham số tốt nhất
grid_search.fit(X_train, y_train)

print(f"Bộ tham số tối ưu nhất: {grid_search.best_params_}")
print(f"Điểm ROC-AUC CV tốt nhất: {grid_search.best_score_:.4f}")
```

---

## 4. Pipeline Thực Chiến Hoàn Chỉnh

Dưới đây là mã nguồn chuẩn hoàn chỉnh trong thực tế: Kết hợp Pipeline, huấn luyện, dự đoán và xuất báo cáo đánh giá chi tiết.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# 1. Tự động sinh dữ liệu giả lập cho bài toán Phân loại
X, y = make_classification(
    n_samples=1500,
    n_features=12,
    n_informative=8,
    n_classes=2,
    weights=[0.7, 0.3],  # Dữ liệu hơi mất cân bằng (70% - 30%)
    random_state=42
)

# Chia tập Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Xây dựng Pipeline
# Lưu ý: Decision Tree KHÔNG yêu cầu Chuẩn hóa dữ liệu (StandardScaler)
pipeline = Pipeline([
    ('model', DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=5,
        class_weight='balanced',  # Tự động điều chỉnh trọng số cho lớp thiểu số
        random_state=42
    ))
])

# 3. Huấn luyện Mô hình
pipeline.fit(X_train, y_train)

# 4. Dự đoán và Đánh giá Performance
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred))
print(f"Chỉ số ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
```
