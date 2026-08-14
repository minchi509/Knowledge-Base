---
topic: "K-Means Clustering"
subtopic: "Optimal K Selection & Diagnostic Metrics"
level: "Intermediate"
doc_id: "km_04"
sources:
  - "Scikit-Learn User Guide: Clustering Performance Evaluation"
key_concepts:
  - "Elbow Method"
  - "Silhouette Coefficient"
  - "Davies-Bouldin Index"
  - "Cluster Visualization"
---

# Diagnostics & Model Evaluation

Do K-Means là thuật toán Unsupervised, ta không có ground-truth labels để tính Accuracy. Ta phải đánh giá chất lượng cụm thông qua các chỉ số cấu trúc hình học.

## 1. Phương pháp Khuỷu tay (Elbow Method)
Vẽ biểu đồ biểu diễn WCSS (Inertia) theo $K$. Khi $K$ tăng, WCSS luôn giảm. Điểm "khuỷu tay" (Elbow point) là vị trí mà tốc độ giảm WCSS suy giảm đột ngột — đại diện cho sự cân bằng giữa độ chính xác và độ phức tạp của mô hình.

## 2. Hệ số Silhouette (Silhouette Coefficient)
 Silhouette đo lường mức độ tương đồng của một mẫu với cụm của chính nó (Cohesion) so với các cụm khác (Separation).
Với mỗi mẫu $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
* $a(i)$: Khoảng cách trung bình từ $i$ đến tất cả các điểm khác trong **cùng cụm**.
* $b(i)$: Khoảng cách trung bình từ $i$ đến tất cả các điểm trong **cụm gần nhất tiếp theo**.

**Thang giá trị**:
* $s(i) \approx +1$: Mẫu được gán rất tốt vào cụm.
* $s(i) \approx 0$: Mẫu nằm sát ranh giới giữa 2 cụm.
* $s(i) \approx -1$: Mẫu bị gán sai cụm.

## 3. Evaluation Script (Elbow + Silhouette Dual Plot)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

def evaluate_kmeans_k_range(X_scaled, k_range=range(2, 11)):
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Đường WCSS (Elbow)
    color = 'tab:blue'
    ax1.set_xlabel('Số cụm (K)')
    ax1.set_ylabel('Inertia (WCSS)', color=color)
    ax1.plot(k_range, inertias, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Đường Silhouette Score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Đánh giá Số cụm K tối ưu (Elbow & Silhouette)')
    plt.grid(True)
    plt.show()
