---
topic: "K-Means Clustering"
subtopic: "Theoretical Foundations & Mathematical Formulation"
level: "Intermediate"
doc_id: "km_01"
source_url: "https://scikit-learn.org/stable/modules/clustering.html#k-means"
key_concepts:
  - "Unsupervised Learning"
  - "Centroids"
  - "Euclidean Distance"
  - "WCSS (Inertia)"
  - "Lloyd's Algorithm"
  - "K-Means++ Initialization"
---

# Theoretical Foundations of K-Means

## 1. Core Concept & Voronoi Partitions
K-Means là thuật toán học không giám sát (Unsupervised Learning) dùng để phân nhóm dữ liệu $X = \{x_1, x_2, ..., x_N\} \in \mathbb{R}^d$ thành $K$ cụm không chồng lấp.
Mỗi cụm $C_j$ được đại diện bởi một **Tâm cụm (Centroid)** $\mu_j$. Thuật toán chia không gian dữ liệu thành các vùng hình học gọi là **Voronoi Cells**, trong đó mỗi điểm dữ liệu thuộc về cụm có tâm gần nhất theo khoảng cách Euclidean.

## 2. Mathematical Objective: WCSS (Inertia)
K-Means giải bài toán tối ưu hóa nhằm tìm tập các tâm cụm $\mu = \{\mu_1, ..., \mu_K\}$ để cực tiểu hóa tổng bình phương khoảng cách nội cụm (**Within-Cluster Sum of Squares - WCSS**), còn gọi là **Inertia**:

$$J(\mu, C) = \sum_{j=1}^{K} \sum_{x_i \in C_j} \Vert{}x_i - \mu_j\Vert{}^2$$

Trong đó:
* $x_i$: Vector đặc trưng của mẫu thứ $i$.
* $\mu_j$: Centroid của cụm $C_j$, được tính bằng trung bình cộng các điểm trong cụm: $\mu_j = \frac{1}{\vert{}C_j\vert{}} \sum_{x_i \in C_j} x_i$.
* $\Vert{}x_i - \mu_j\Vert{}^2$: Bình phương khoảng cách Euclidean $d(x, \mu)^2 = \sum_{m=1}^{d} (x_{im} - \mu_{jm})^2$.

## 3. Lloyd's Algorithm & Convergence
Thuật toán Lloyd giải bài toán tối ưu $J(\mu, C)$ bằng chiến lược lặp hai bước (Expectation-Maximization):

1. **Assignment Step (Bước E)**: Gán từng điểm $x_i$ vào cụm $C_j^{(t)}$ có tâm gần nhất:
   $$C_j^{(t)} = \{ x_i : \Vert{}x_i - \mu_j^{(t)}\Vert{}^2 \le \Vert{}x_i - \mu_l^{(t)}\Vert{}^2, \, \forall l, 1 \le l \le K \}$$
2. **Update Step (Bước M)**: Tính lại vị trí các tâm cụm dựa trên các điểm vừa gán:
   $$\mu_j^{(t+1)} = \frac{1}{\vert{}C_j^{(t)}\vert{}} \sum_{x_i \in C_j^{(t)}} x_i$$
3. **Điều kiện dừng**: Thuật toán dừng khi $\Vert{} \mu^{(t+1)} - \mu^{(t)} \Vert{} < \epsilon$ hoặc đạt số vòng lặp tối đa `max_iter`.

## 4. K-Means++ Initialization Math
Khởi tạo tâm ngẫu nhiên dễ khiến K-Means bẫy vào cực tiểu cục bộ (Local Minima). Thuật toán **K-Means++** giải quyết bằng cách chọn các tâm ban đầu cách xa nhau:
1. Chọn ngẫu nhiên tâm đầu tiên $\mu_1$ từ tập dữ liệu.
2. Với mỗi điểm $x_i$, tính khoảng cách ngắn nhất $D(x_i)$ đến tâm gần nhất đã chọn.
3. Chọn tâm tiếp theo $\mu_k$ từ tập dữ liệu với xác suất tỷ lệ thuận với $D(x_i)^2$:
   $$P(x_i) = \frac{D(x_i)^2}{\sum_{l=1}^{N} D(x_l)^2}$$
4. Lặp lại bước 2-3 cho đến khi đủ $K$ tâm.

## 5. From-Scratch Implementation (NumPy)

```python
import numpy as np

def kmeans_single_step_scratch(X, centroids):
    """
    Thực hiện 1 vòng lặp Lloyd: Assignment và Update Centroids
    """
    # 1. Assignment Step: Tính ma trận khoảng cách (N, K)
    # Broadcasting: (N, 1, D) - (1, K, D) -> (N, K, D)
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    
    # 2. Update Step: Tính lại tâm cụm
    K = centroids.shape[0]
    new_centroids = np.zeros_like(centroids)
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
        else:
            new_centroids[k] = centroids[k] # Giữ nguyên nếu cụm rỗng
            
    # Tính WCSS (Inertia)
    inertia = np.sum((X - new_centroids[labels]) ** 2)
    
    return new_centroids, labels, inertia
