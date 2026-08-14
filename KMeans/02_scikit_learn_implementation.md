---
topic: "K-Means Clustering"
subtopic: "Scikit-Learn API & Hyperparameters"
level: "Intermediate"
doc_id: "km_02"
sources:
  - "Scikit-Learn API: sklearn.cluster.KMeans"
key_concepts:
  - "n_clusters"
  - "init"
  - "n_init"
  - "max_iter"
  - "tol"
---

# Scikit-Learn API & Hyperparameter Matrix

## 1. Class Specification
Lớp chính: `sklearn.cluster.KMeans`.
* `.fit(X)`: Tìm vị trí tâm cụm tối ưu từ dữ liệu $X$.
* `.fit_predict(X)`: Huấn luyện và trả về nhãn cụm cho tập $X$.
* `.transform(X)`: Biến đổi $X$ thành không gian $K$ chiều, đại diện cho khoảng cách từ từng mẫu đến $K$ tâm cụm.
* `.cluster_centers_`: Numpy array shape `(n_clusters, n_features)` chứa tọa độ các centroid.
* `.inertia_`: Giá trị WCSS tối ưu đạt được.

## 2. Hyperparameter Configuration Matrix

| Tham số | Kiểu dữ liệu | Mặc định | Mức độ ảnh hưởng & Hướng dẫn cấu hình |
| :--- | :--- | :--- | :--- |
| `n_clusters` | `int` | `8` | Số cụm $K$. Tham số quan trọng nhất. Cần kết hợp Elbow Method và Silhouette Analysis để chọn. |
| `init` | `str` / `array` | `'k-means++'` | Phương pháp khởi tạo tâm: `'k-means++'`, `'random'` hoặc mảng tọa độ ban đầu. **Luôn dùng `'k-means++'`**. |
| `n_init` | `int` / `str` | `'auto'` | Số lần chạy thuật toán độc lập với các hạt giống khác nhau. Bản chạy có `inertia_` thấp nhất sẽ được giữ lại. (Khuyên dùng $10$). |
| `max_iter` | `int` | `300` | Số vòng lặp tối đa trong 1 lần chạy. Nếu mô hình chưa hội tụ, tăng lên $500$. |
| `tol` | `float` | `1e-4` | Ngưỡng dung sai về sự thay đổi vị trí tâm cụm để xác định thuật toán đã hội tụ. |
| `algorithm` | `str` | `'lloyd'` | Algorithmic variant: `'lloyd'` hoặc `'elkan'`. `'elkan'` nhanh hơn trên dữ liệu có cụm rõ ràng nhờ bất đẳng thức tam giác. |

## 3. Production Executable Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# 1. Tạo dữ liệu giả lập 3 cụm
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42)

# 2. Định nghĩa Production Pipeline với Scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=3,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42
    ))
])

# 3. Fit và trích xuất thông tin
labels = pipeline.fit_predict(X)
kmeans_model = pipeline.named_steps['kmeans']

print(f"WCSS (Inertia): {kmeans_model.inertia_:.2f}")
print(f"Số vòng lặp hội tụ: {kmeans_model.n_iter_}")
print("Tọa độ Centroid (đã Scale):")
print(kmeans_model.cluster_centers_)
