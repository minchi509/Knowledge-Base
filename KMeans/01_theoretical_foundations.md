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
---

# Theoretical Foundations of K-Means

## 1. Unsupervised Learning Concept
Khác với Logistic Regression hay Decision Tree (Học có giám sát - Supervised Learning), K-Means là thuật toán **Học không giám sát (Unsupervised Learning)**. Trong bộ dữ liệu như *Mall Customer Segmentation*, chúng ta không có nhãn (label) cho biết khách hàng thuộc nhóm nào. Nhiệm vụ của K-Means là tự động gom cụm (clustering) các điểm dữ liệu có đặc tính giống nhau thành $K$ nhóm riêng biệt.

## 2. Objective Function: WCSS (Inertia)
K-Means tìm cách phân chia tập dữ liệu $X$ thành $K$ cụm $C = \{C_1, C_2, ..., C_K\}$ sao cho tổng bình phương khoảng cách từ mỗi điểm dữ liệu đến tâm cụm (centroid) của nó là nhỏ nhất. Đại lượng này gọi là **Within-Cluster Sum of Squares (WCSS)** hay **Inertia**:

$$J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \vert{}\vert{}x_i - \mu_j\vert{}\vert{}^2$$

Trong đó:
* $x_i$: Điểm dữ liệu thứ $i$.
* $\mu_j$: Tâm (centroid) của cụm $C_j$.
* $\vert{}\vert{}x_i - \mu_j\vert{}\vert{}^2$: Bình phương khoảng cách Euclidean giữa điểm dữ liệu và tâm cụm.

## 3. Lloyd's Algorithm (Quy trình huấn luyện)
Thuật toán K-Means tối ưu hóa hàm $J$ thông qua vòng lặp lặp đi lặp lại 2 bước (Expectation-Maximization logic):
1. **Khởi tạo (Initialization)**: Chọn ngẫu nhiên $K$ điểm làm tâm cụm ban đầu.
2. **Gán cụm (Assignment)**: Gán mỗi điểm dữ liệu $x_i$ vào cụm có tâm $\mu_j$ gần nó nhất.
3. **Cập nhật tâm (Update)**: Tính lại vị trí của tâm $\mu_j$ bằng cách lấy trung bình cộng tọa độ của tất cả các điểm vừa được gán vào cụm $C_j$.
4. **Lặp lại (Repeat)**: Lặp lại bước 2 và 3 cho đến khi các tâm cụm không thay đổi vị trí (thuật toán hội tụ).

## 4. From-Scratch Implementation (NumPy)
Để học viên hiểu bản chất bước gán cụm (Assignment) bằng khoảng cách Euclidean:

```python
import numpy as np

def assign_clusters_scratch(X, centroids):
    """
    Hàm tính khoảng cách Euclidean và gán cụm thủ công
    """
    # X có shape (n_samples, n_features)
    # centroids có shape (K, n_features)
    
    # Tính khoảng cách từ mỗi điểm đến tất cả các centroid
    # Sử dụng broadcasting của NumPy
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    
    # Lấy index của centroid gần nhất (khoảng cách nhỏ nhất)
    cluster_labels = np.argmin(distances, axis=1)
    
    return cluster_labels
