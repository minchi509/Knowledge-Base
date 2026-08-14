---
topic: "K-Means Clustering"
subtopic: "Scikit-Learn API & Hyperparameters"
level: "Intermediate"
doc_id: "km_02"
sources:
  - "Scikit-Learn API: sklearn.cluster.KMeans"
key_concepts:
  - "n_clusters"
  - "init='k-means++'"
  - "n_init"
  - "max_iter"
---

# Scikit-Learn API & Hyperparameters

## 1. Main Class Specification
Lớp chính: `sklearn.cluster.KMeans`.
* `.fit(X)`: Tìm các tâm cụm trên dữ liệu huấn luyện. (Không có $y$).
* `.predict(X)`: Dự đoán cụm cho các mẫu dữ liệu mới.
* `.fit_predict(X)`: Huấn luyện và trả về nhãn cụm cho chính tập $X$.
* `.cluster_centers_`: Thuộc tính chứa tọa độ của $K$ tâm cụm sau khi hội tụ.
* `.inertia_`: Giá trị WCSS cuối cùng.

## 2. Hyperparameter Configuration Matrix

| Tham số | Mặc định | Ý nghĩa & Lời khuyên tối ưu |
| :--- | :--- | :--- |
| `n_clusters` | `8` | Số lượng cụm $K$. Đây là tham số quan trọng nhất, cần xác định bằng thuật toán (Elbow/Silhouette) chứ không đoán mò. |
| `init` | `'k-means++'` | Thuật toán khởi tạo tâm cụm. Mặc định là `'k-means++'` giúp các tâm cụm ban đầu cách xa nhau, tránh rơi vào cực tiểu cục bộ (Random Initialization Trap). **Luôn giữ mặc định này**. |
| `n_init` | `'auto'` | Số lần thuật toán chạy độc lập với các khởi tạo ngẫu nhiên khác nhau. Kết quả trả về là lần chạy có `inertia_` thấp nhất. (Nên đặt $>10$ nếu không dùng k-means++). |
| `max_iter` | `300` | Số vòng lặp tối đa cho một lần chạy. K-Means thường hội tụ rất nhanh, $300$ là quá đủ. |
| `random_state` | `None` | Cố định hạt giống ngẫu nhiên để đảm bảo tính tái lập kết quả khi sinh Notebook bài tập. |

## 3. Implementation Example

```python
from sklearn.cluster import KMeans
import numpy as np

# Giả lập dữ liệu khách hàng (Thu nhập, Điểm chi tiêu)
X = np.random.rand(100, 2) * 100 

# Khởi tạo mô hình
kmeans = KMeans(
    n_clusters=5, 
    init='k-means++', 
    n_init='auto', 
    random_state=42
)

# Huấn luyện và gán nhãn
cluster_labels = kmeans.fit_predict(X)

print("Tọa độ tâm cụm (Centroids):")
print(kmeans.cluster_centers_)
print(f"Tổng bình phương khoảng cách (Inertia): {kmeans.inertia_:.2f}")
