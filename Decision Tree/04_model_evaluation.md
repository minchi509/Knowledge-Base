---
topic: "Decision Tree"
subtopic: "Tree Diagnostics & Interpretability"
level: "Intermediate"
doc_id: "dt_04"
sources:
  - "Scikit-Learn User Guide v1.4: Tree Visualization"
key_concepts:
  - "plot_tree"
  - "export_text"
  - "Decision Path"
---

# Tree Diagnostics & Interpretability

## 1. Sức mạnh Diễn giải (Interpretability)
Lợi thế cạnh tranh lớn nhất của Decision Tree so với Deep Learning là khả năng minh bạch hoàn toàn (White-box Model). Bạn có thể giải thích chính xác tại sao một mẫu dữ liệu lại được phân vào lớp A thay vì lớp B.

### 1.1. Trực quan hóa cấu trúc bằng `plot_tree`
Sử dụng thư viện matplotlib để vẽ cây. Phù hợp cho việc trình bày với cấp quản lý.

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def visualize_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,          # Nền nút được tô màu dựa trên lớp chiếm đa số
        rounded=True,         # Bo tròn viền nút
        proportion=False,     # Hiển thị số lượng mẫu cụ thể (True sẽ hiển thị %)
        fontsize=10,
        max_depth=3           # Cắt hiển thị ở tầng 3 để ảnh không bị nhòe
    )
    plt.title("Cấu trúc Cây Quyết Định", fontsize=18)
    plt.show()
```
### 1.2. Xuất luật quyết định bằng export_text
Đôi khi việc nhúng các luật (Rules) vào hệ thống backend (Java/C#) yêu cầu định dạng văn bản thô.

Python
from sklearn.tree import export_text
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
```
# Output ví dụ:
# |--- Tuổi <= 30.50
# |   |--- Thu_nhập <= 1500.00
# |   |   |--- class: 0
# |   |--- Thu_nhập >  1500.00
# |   |   |--- class: 1
```

## 2. Truy vết đường dẫn quyết định (Decision Path)
Scikit-learn cung cấp hàm decision_path(X) để theo dõi một bệnh nhân/khách hàng cụ thể đã đi qua những nút (nút điều kiện) nào trước khi đạt tới quyết định cuối cùng.

Python
Lấy ra đường dẫn cho mẫu dữ liệu đầu tiên
sample_id = 0
node_indicator = model.decision_path(X_test)
leave_id = model.apply(X_test)

node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print(f"Luật suy luận cho mẫu {sample_id}:")
for node_id in node_index:
    if leave_id[sample_id] == node_id:
        print(f"--> Đạt tới Nút lá {node_id}.")
        continue
    # Các điều kiện đi qua
    print(f"Đi qua nút {node_id}")
