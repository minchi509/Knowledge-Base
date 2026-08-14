---
topic: "K-Means Clustering"
subtopic: "Data Preprocessing, Feature Scaling & Dimensionality Reduction"
level: "Intermediate"
doc_id: "km_03"
sources:
  - "Scikit-Learn User Guide: Preprocessing"
key_concepts:
  - "Euclidean Distance Distortion"
  - "StandardScaler"
  - "Outlier Sensitivity"
  - "PCA Integration"
---

# Preprocessing & Dimensionality Reduction Guidelines

## 1. Mức độ nhạy cảm với Thang đo (Scaling Sensitivity)
K-Means dựa hoàn toàn vào khoảng cách Euclidean $d(x, y) = \sqrt{\sum (x_i - y_i)^2}$.
Nếu một đặc trưng có biên độ lớn (ví dụ `Annual Income` từ $15,000 - $130,000) và đặc trưng khác có biên độ nhỏ (ví dụ `Age` từ $18 - 70$), khoảng cách Euclidean sẽ bị chi phối $99.9\%$ bởi `Annual Income`. `Age` gần như không có đóng góp gì vào việc phân cụm.

**Bắt buộc**: Luôn chuẩn hóa dữ liệu bằng `StandardScaler` trước khi đưa vào K-Means.

## 2. Ảnh hưởng của Outliers
Vì WCSS tính **bình phương** khoảng cách $\Vert{}x_i - \mu_j\Vert{}^2$, một điểm dữ liệu ngoại lệ nằm cực xa sẽ đóng góp giá trị WCSS rất lớn. Thuật toán sẽ bị ép phải kéo centroid về phía điểm ngoại lệ đó, làm sai lệch cấu trúc của toàn bộ cụm.
* **Xử lý**: Loại bỏ outliers bằng IQR/Z-score trước khi gom cụm, hoặc đổi sang `DBSCAN` nếu dữ liệu có nhiều nhiễu.

## 3. Kết hợp PCA khi dữ liệu nhiều chiều (High-Dimensional Data)
Khi dữ liệu có $>10$ đặc trưng, khoảng cách Euclidean chịu hiện tượng **Lời nguyền chiều dữ liệu (Curse of Dimensionality)**: khoảng cách giữa mọi cặp điểm dần tiến về bằng nhau, khiến K-Means mất khả năng phân cụm.
Giải pháp chuẩn: Dùng **PCA (Principal Component Analysis)** để giảm chiều dữ liệu về 2-5 thành phần chính trước khi gom cụm.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Pipeline đầy đủ: Scale -> PCA (Giữ 90% phương sai) -> K-Means
complete_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.90, random_state=42)),
    ('kmeans', KMeans(n_clusters=5, init='k-means++', random_state=42))
])

# Fit pipeline trên dữ liệu Mall Customers đa chiều
# df_features gồm: Age, Annual Income, Spending Score, Male_binary
cluster_labels = complete_pipeline.fit_predict(X_multi_dim)
print(f"Số chiều sau PCA: {complete_pipeline.named_steps['pca'].n_components_}")
