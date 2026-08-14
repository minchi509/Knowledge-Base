---
topic: "Decision Tree"
subtopic: "Data Preprocessing, Encoding & Feature Importance"
level: "Intermediate"
doc_id: "dt_03"
sources:
  - "Scikit-Learn User Guide v1.4: Trees Preprocessing"
key_concepts:
  - "Scale Invariance"
  - "High Cardinality Bias"
  - "Ordinal vs One-Hot Encoding"
  - "Feature Importance (Gini Importance)"
---

# Data Preprocessing & Feature Engineering Guidelines

## 1. Scale Invariance (Bất biến với Thang đo)
Thuộc tính độc đáo nhất của Decision Tree: **Không yêu cầu chuẩn hóa dữ liệu**.
* Các thuật toán như Logistic Regression (dùng Gradient Descent) hay KNN (dùng khoảng cách Euclidean) sẽ bị sai lệch nếu một biến tính bằng Kilomet và biến kia tính bằng Milimet.
* Decision Tree hoạt động bằng cách sắp xếp giá trị của từng cột một cách độc lập và dò tìm điểm cắt ngưỡng (threshold). Các phép biến đổi đơn điệu (như nhân, chia, logarit) không làm thay đổi thứ tự tương đối của các điểm dữ liệu, do đó cấu trúc cây và kết quả phân loại giữ nguyên vẹn 100%.

## 2. Categorical Encoding & High Cardinality Bias
Scikit-learn yêu cầu đầu vào dạng số nguyên hoặc số thực. Tuy nhiên, việc áp dụng mã hóa nào ảnh hưởng rất lớn đến cây:

* **One-Hot Encoding**: Thường được dùng cho biến Nominal.
  * *Hậu quả*: Nó tạo ra ma trận dữ liệu rất thưa thớt (sparse). Cây quyết định mắc phải hội chứng **Cardinality Bias** (Thiên vị kích thước tập giá trị). Nó luôn ưu tiên phân nhánh ở các biến liên tục (như `Tuổi`, có rất nhiều điểm cắt để dò) và bỏ qua các biến One-Hot (chỉ có cắt tại $0.5$).
* **Ordinal Encoding**: Đối với Tree-based models, dùng `OrdinalEncoder` (như $1, 2, 3...$) thường mang lại kết quả tốt hơn hoặc tương đương One-Hot, đồng thời giúp cây không bị nổ kích thước (explosion of depth).

## 3. Extracting Feature Importance (Gini Importance)
Sau khi `fit()`, mô hình cung cấp thuộc tính `feature_importances_`.
**Cơ sở toán học**: Độ quan trọng của một đặc trưng $j$ được tính bằng tổng lượng Information Gain (hoặc Gini Decrease) mà đặc trưng đó đóng góp trên toàn bộ các nút của cây, được chuẩn hóa sao cho tổng các độ quan trọng của mọi đặc trưng bằng $1.0$.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_tree_feature_importance(model, feature_names):
    """
    Trích xuất và vẽ biểu đồ mức độ đóng góp của từng Feature.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Sắp xếp giảm dần
    
    plt.figure(figsize=(10, 6))
    plt.title("Mức độ quan trọng của các Đặc trưng (Gini Importance)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
