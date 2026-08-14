---
topic: "K-Means Clustering"
subtopic: "Determining Optimal K & Model Diagnostics"
level: "Intermediate"
doc_id: "km_04"
sources:
  - "Scikit-Learn User Guide: Clustering Evaluation"
key_concepts:
  - "Elbow Method"
  - "Silhouette Score"
  - "Visualization"
---

# Diagnostics & Model Evaluation

Vì là học không giám sát (không có nhãn đúng để so sánh), ta không thể dùng Accuracy hay F1-Score. Để đánh giá chất lượng cụm và tìm số $K$ tối ưu, ta dùng hai phương pháp sau:

## 1. Phương pháp Khuỷu tay (The Elbow Method)

Ghi nhận giá trị **Inertia (WCSS)** khi tăng dần số cụm $K$ (từ 1 đến 10). Khi $K$ tăng, Inertia luôn giảm. Tuy nhiên, tại một điểm $K$ nào đó, tốc độ giảm sẽ chậm lại đột ngột tạo thành hình "khuỷu tay". Đó chính là số cụm tối ưu.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('Phương pháp Khuỷu tay (Elbow Method)')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()
```

## 2. Hệ số Silhouette (Silhouette Score)

Silhouette đo lường hai yếu tố:

1. **Cohesion ($a$)**: Điểm dữ liệu gần với các điểm khác trong cùng cụm như thế nào.
2. **Separation ($b$)**: Điểm dữ liệu cách xa các cụm khác như thế nào.

$$S = \frac{b - a}{\max(a, b)}$$

Giá trị $S \in [-1, 1]$. $S$ càng gần $1$, cụm phân chia càng rõ ràng và chất lượng càng cao.

```python
from sklearn.metrics import silhouette_score

# Tính Silhouette cho mô hình K=5
score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score (K=5): {score:.4f}")
```

## 3. Trực quan hóa Cụm (Visualization)

Đối với bài toán *Mall Customer*, nếu ta lấy 2 biến `Annual Income` và `Spending Score` để chạy K-Means, ta có thể vẽ biểu đồ 2D để xem các cụm:

```python
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=cluster_labels,
    cmap='viridis'
)

# Lấy tọa độ tâm cụm từ pipeline
centroids = clustering_pipeline.named_steps['kmeans'].cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments')
plt.legend()
plt.show()
```
