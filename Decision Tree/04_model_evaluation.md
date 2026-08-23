---
topic: "Decision Tree"
subtopic: "Tree Diagnostics & Interpretability"
level: "Intermediate"
doc_id: "dt_04"
source_url: "https://scikit-learn.org/stable/modules/tree.html#tree-visualization"
key_concepts:
  - "Hàm plot_tree (Vẽ cấu trúc cây)"
  - "Hàm export_text (Xuất luật dạng văn bản)"
  - "Hàm decision_path (Truy vết đường dẫn quyết định)"
  - "Mô hình hộp trắng (White-box Model)"
  - "Khả năng diễn giải mô hình (Interpretability)"
---

# Chẩn Đoán Cây & Khả Năng Diễn Giải Mô Hình

## 1. Sức Mạnh Diễn Giải & Tính Minh Bạch (Model Interpretability)

Ưu điểm vượt trội nhất của Decision Tree (Cây quyết định) so với các mô hình như Neural Networks hay Gradient Boosting là tính **Minh bạch hoàn toàn (White-box Model)**.

- **White-box vs Black-box:** Mô hình Hộp đen (Black-box) đưa ra dự đoán mà không giải thích rõ nguyên do. Trong khi đó, Cây quyết định cung cấp chuỗi logic IF-THEN rõ ràng, giúp chuyên gia domain (bác sĩ, chuyên viên tài chính) dễ dàng thẩm định và tin tưởng kết quả.
- **Ứng dụng thực tế:** Phù hợp cho các lĩnh vực đòi hỏi tính giải trình cao như Tín dụng (giải thích lý do từ chối vay) và Y tế (giải thích lý do chẩn đoán bệnh).

---

## 2. Trực Quan Hóa Và Trích Xuất Luật Quyết Định (Decision Rules)

Scikit-Learn cung cấp 2 công cụ chính để xem cấu trúc bên trong của Cây quyết định:

### 2.1. Vẽ cấu trúc đồ họa bằng `plot_tree`

Hàm `plot_tree` kết hợp với thư viện `matplotlib` cho phép hiển thị sơ đồ cây trực quan, tô màu theo độ thuần khiết của nút.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Tải dữ liệu và huấn luyện mô hình đơn giản
iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# 2. Vẽ cấu trúc cây
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,          # Tô màu nút dựa trên lớp chiếm đa số
    rounded=True,          # Bo tròn khung nút
    impression=False,      # Hiển thị tỷ lệ thay vì số mẫu tuyệt đối
    fontsize=10
)
plt.title("Sơ đồ Cấu trúc Cây Quyết định (Iris Dataset)", fontsize=14)
plt.show()
```

**Các siêu tham số cần nhớ khi dùng `plot_tree`:**

- `filled=True`: Mức độ đậm nhạt của màu sắc phản ánh độ thuần khiết (Impurity càng nhỏ màu càng đậm).
- `max_depth`: Giới hạn số tầng hiển thị khi cây quá lớn, tránh tình trạng hình ảnh bị nhòe hoặc đè chữ.

### 2.2. Trích xuất dạng văn bản thô bằng `export_text`

Khi cần nhúng chuỗi quy tắc (Rules) trực tiếp vào hệ thống backend (viết lại bằng Java, C#, C++) hoặc lưu log, ta dùng `export_text`.

```python
from sklearn.tree import export_text

# Trích xuất luật dưới dạng cây thư mục text
tree_rules = export_text(clf, feature_names=list(iris.feature_names))
print("=== CHUỖI LUẬT SUY LUẬN TỪ CÂY ===")
print(tree_rules)
```

**Output mẫu thực tế:**

```
=== CHUỖI LUẬT SUY LUẬN TỪ CÂY ===
|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal length (cm) <= 4.75
|   |   |--- class: 1
|   |--- petal length (cm) >  4.75
|   |   |--- class: 2
```

---

## 3. Truy Vết Đường Dẫn Quyết Định Cho Từng Mẫu Cụ Thể (Decision Path)

Đôi khi ta cần giải thích cho riêng một khách hàng hay một bệnh nhân cụ thể: "Tại sao hệ thống lại từ chối hồ sơ của vị khách này?".

Scikit-Learn cung cấp hai hàm hỗ trợ việc này:

- `model.apply(X)`: Trả về ID của nút lá mà mẫu $X$ rơi vào.
- `model.decision_path(X)`: Trả về ma trận thưa (Sparse Matrix) thể hiện chuỗi tất cả các nút (từ gốc đến lá) mà mẫu $X$ đã đi qua.

**Mã nguồn truy vết quy tắc cho 1 mẫu dữ liệu cụ thể:**

```python
import numpy as np

# Chọn mẫu dữ liệu cần phân tích (Mẫu đầu tiên trong tập dữ liệu)
sample_id = 0
sample_data = X[sample_id : sample_id + 1]  # Dạng shape (1, n_features)

# 1. Lấy chỉ số nút lá và ma trận đường đi
node_indicator = clf.decision_path(sample_data)
leaf_id = clf.apply(sample_data)[0]

# 2. Trích xuất danh sách ID của các nút mà mẫu này đi qua
node_index = node_indicator.indices[
    node_indicator.indptr[0] : node_indicator.indptr[1]
]

print(f"=== GIẢI THÍCH QUY TRÌNH RA QUYẾT ĐỊNH CHO MẪU {sample_id} ===")
print(f"Giá trị đặc trưng của mẫu: {X[sample_id]}")
print(f"Kết quả dự đoán (Lớp): {clf.predict(sample_data)[0]} ({iris.target_names[clf.predict(sample_data)[0]]})\n")

# Duyệt qua từng nút trên đường đi để in câu hỏi điều kiện
feature = clf.tree_.feature
threshold = clf.tree_.threshold

for node_id in node_index:
    # Nếu là nút lá, dừng truy vết
    if leaf_id == node_id:
        print(f"--> Nút {node_id} [LÁ]: Đạt quyết định cuối cùng.")
        break

    # Kiểm tra điều kiện phân nhánh tại nút nội bộ
    node_feature = feature[node_id]
    feature_name = iris.feature_names[node_feature]
    node_threshold = threshold[node_id]

    # Rẽ trái nếu <= threshold, ngược lại rẽ phải
    if sample_data[0][node_feature] <= node_threshold:
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print(
        f"Tại Nút {node_id}: {feature_name} (Giá trị = {sample_data[0][node_feature]:.2f}) "
        f"{threshold_sign} {node_threshold:.2f} -> Thỏa mãn, đi tiếp."
    )
```

---

## 4. Tóm Tắt & Quy Tắc Kiểm Định Mô Hình (Diagnostics Checklist)

- **Cắt tỉa cây trước khi vẽ:** Luôn đặt `max_depth` (thường $\le 4$) trước khi dùng `plot_tree`. Nếu cây quá sâu (độ sâu > 10), hình vẽ biểu đồ sẽ bị đè chữ và hoàn toàn không thể đọc được.
- **Ưu tiên `export_text` cho cây phức tạp:** Khi cây có quá nhiều nhánh, chuyển sang `export_text` để dễ tìm kiếm (Ctrl+F) hoặc đọc dạng văn bản.
- **Độc lập nền tảng nhờ `decision_path`:** Dùng `decision_path` để xây dựng tính năng "Giải thích lý do dự đoán" (Explainable AI) trên giao diện ứng dụng thực tế cho người dùng cuối.
