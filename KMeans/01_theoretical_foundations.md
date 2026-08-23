---
topic: "K-Means Clustering"
subtopic: "Theoretical Foundations & Mathematical Formulation"
level: "Intermediate"
doc_id: "km_01"
source_url: "https://scikit-learn.org/stable/modules/clustering.html#k-means"
key_concepts:
  - "Học không giám sát (Unsupervised Learning)"
  - "Tâm cụm & Phân hoạch Voronoi (Centroids & Voronoi Partitions)"
  - "Tổng bình phương khoảng cách nội cụm WCSS (Inertia)"
  - "Thuật toán Lloyd (Lloyd's Algorithm - EM)"
  - "Khởi tạo K-Means++ (K-Means++ Initialization)"
---

# Nền Tảng Lý Thuyết của K-Means (Theoretical Foundations of K-Means)

## 1. Khái niệm Cốt lõi & Phân hoạch Voronoi (Voronoi Partitions)

**K-Means** là một trong những thuật toán Học không giám sát (Unsupervised Learning) phổ biến nhất, được thiết kế để phân chia tập dữ liệu chưa có nhãn $X = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^d$ thành $K$ cụm (clusters) riêng biệt và không chồng lấp: $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$.

### Trực quan hóa hình học: Ô Voronoi (Voronoi Cells)

* **Tâm cụm (Centroid):** Mỗi cụm $C_j$ được đại diện bởi một điểm trung tâm $\mu_j \in \mathbb{R}^d$.
* **Quy tắc phân vùng:** Một điểm dữ liệu $x_i$ thuộc về cụm $C_j$ nếu khoảng cách từ $x_i$ đến tâm $\mu_j$ là ngắn nhất so với tất cả các tâm khác.
* **Tế bào Voronoi (Voronoi Partitioning):** Ranh giới giữa các cụm tạo thành các đường trung trực (trong không gian 2D) hoặc các siêu phẳng (hyperplanes trong không gian nhiều chiều). Tập hợp các vùng này chia không gian đặc trưng thành các **Voronoi Cells**.

$$\text{Mẫu } x_i \in C_j \iff \| x_i - \mu_j\| ^2 \le \| x_i - \mu_l\| ^2 \quad \forall l \in \{1, \dots, K\}$$

---

## 2. Hàm Mục tiêu Toán học: WCSS (Inertia)

Thuật toán K-Means phát biểu bài toán phân cụm dưới dạng một **bài toán tối ưu hóa toán học**. Mục tiêu là tìm tập hợp các tâm cụm $\boldsymbol{\mu} = \{\mu_1, \mu_2, \dots, \mu_K\}$ và các tập gán cụm $C$ sao cho tổng khoảng cách từ các điểm đến tâm cụm của chúng là nhỏ nhất.

### 2.1. Công thức WCSS (Within-Cluster Sum of Squares)

Hàm mất mát (Loss function) của K-Means, còn gọi là **Inertia** hoặc **WCSS**, được định nghĩa:

$$J(\boldsymbol{\mu}, \mathcal{C}) = \sum_{j=1}^{K} \sum_{x_i \in C_j} \| x_i - \mu_j\| ^2$$

Trong đó:

* $N$: Tổng số lượng mẫu dữ liệu ($i = 1, \dots, N$).
* $K$: Số lượng cụm do người dùng chỉ định ($j = 1, \dots, K$).
* $x_i \in \mathbb{R}^d$: Vector đặc trưng $d$-chiều của điểm dữ liệu thứ $i$.
* $\mu_j \in \mathbb{R}^d$: Vector tâm của cụm $C_j$.
* $\| x_i - \mu_j\| ^2$: Bình phương khoảng cách Euclidean giữa điểm $x_i$ và tâm $\mu_j$:

$$\| x_i - \mu_j\| ^2 = \sum_{m=1}^{d} (x_{im} - \mu_{jm})^2$$

### 2.2. Tính chất của Hàm Mục tiêu $J$

1. **Độ tập trung (Compactness):** Giá trị $J$ càng nhỏ, các điểm trong cùng một cụm càng nằm gần tâm, thể hiện độ thu gọn của cụm càng cao.
2. **Quan hệ với $K$:** Khi tăng $K$, hàm $J$ luôn giảm đơn điệu. Đặc biệt khi $K = N$, $J = 0$ (mỗi điểm là một cụm). Do đó, không thể dùng $J$ đơn lẻ để chọn $K$ tối ưu.
3. **Bài toán NP-Hard:** Việc tìm đáp án tối ưu toàn cục (Global Optimum) cho $J$ là một bài toán NP-hard. Thuật toán K-Means sử dụng phương pháp heuristic để tìm **cực tiểu cục bộ (Local Minimum)**.

---

## 3. Thuật toán Lloyd (K-Means Iteration) & Sự hội tụ

Thuật toán chuẩn của K-Means (do Stuart Lloyd đề xuất) giải bài toán tối ưu $J(\boldsymbol{\mu}, \mathcal{C})$ bằng chiến lược lặp hai bước theo nguyên lý **Expectation-Maximization (EM)**.

### Quy trình lặp 3 bước:

```text
[BƯỚC 1: Khởi tạo] ----> Chọn K tâm cụm ban đầu μ_1, μ_2, ..., μ_K
        |
        v
[BƯỚC 2: Assignment (E-Step)]  Cố định μ, gán mỗi điểm x_i vào cụm có tâm gần nhất
        |
        v
[BƯỚC 3: Update (M-Step)]      Cố định cụm, tính lại tâm μ_j = trung bình cộng các x_i
        |
        +-----> [Kiểm tra Hội tụ] -> Nếu μ không đổi (hoặc ΔJ < ε) -> DỪNG
                Nếu chưa hội tụ    -> Lặp lại BƯỚC 2
```

### Chi tiết Toán học từng bước:

**Assignment Step (Bước E - Gán cụm):** Cố định vị trí các tâm $\mu_j^{(t)}$, tối ưu hóa việc phân chia tập cụm $C^{(t)}$:

$$C_j^{(t)} = \left\{ x_i : \left\|  x_i - \mu_j^{(t)} \right\| ^2 \le \left\|  x_i - \mu_l^{(t)} \right\| ^2 \, \forall l, 1 \le l \le K \right\}$$

**Update Step (Bước M - Cập nhật tâm):** Cố định tập cụm $C_j^{(t)}$, tính lại tâm $\mu_j^{(t+1)}$ bằng cách lấy đạo hàm hàm $J$ theo $\mu_j$ và cho bằng $0$:

$$\frac{\partial J}{\partial \mu_j} = -2 \sum_{x_i \in C_j} (x_i - \mu_j) = 0 \implies \mu_j^{(t+1)} = \frac{1}{| C_j^{(t)}| } \sum_{x_i \in C_j^{(t)}} x_i$$

(Tâm cụm chính là trung bình cộng (Mean) của tất cả các vector dữ liệu thuộc cụm đó).

**Chứng minh Sự hội tụ (Convergence Guarantee):**

* Trong mỗi bước (E hoặc M), giá trị hàm $J$ luôn giảm hoặc giữ nguyên: $J^{(t+1)} \le J^{(t)}$.
* Số lượng cách phân chia $N$ điểm vào $K$ cụm là hữu hạn ($K^N$ cách).
* Do đó, thuật toán Lloyd bắt buộc phải hội tụ sau một số bước lặp hữu hạn (thường rất nhanh trong thực tế).

---

## 4. Khởi tạo Tâm cụm nâng cao: K-Means++

Khởi tạo tâm ngẫu nhiên hoàn toàn (Random Initialization) khiến K-Means dễ rơi vào các bẫy cực tiểu cục bộ xấu (bad local minima), phụ thuộc lớn vào may rủi.

Thuật toán K-Means++ giải quyết triệt để vấn đề này bằng cơ chế chọn các tâm ban đầu phân tán xa nhau nhất có thể.

```text
[Tâm 1] --(Chọn xa nhất có thể)--> [Tâm 2] --(Chọn xa các tâm hiện có)--> [Tâm 3]
```

### Các bước thực hiện K-Means++:

1. Chọn ngẫu nhiên tâm đầu tiên $\mu_1$ từ tập dữ liệu $X$ theo phân phối đều.
2. Với mỗi điểm dữ liệu $x_i$, tính khoảng cách ngắn nhất từ nó đến tâm gần nhất đã được chọn:

$$D(x_i) = \min_{j \in \{1, \dots, k\}} \| x_i - \mu_j\| $$

3. Chọn điểm tiếp theo $\mu_{k+1}$ từ $X$ với xác suất $P(x_i)$ tỷ lệ thuận với bình phương khoảng cách $D(x_i)^2$:

$$P(x_i) = \frac{D(x_i)^2}{\sum_{l=1}^{N} D(x_l)^2}$$

4. Lặp lại bước 2 và 3 cho đến khi đã chọn đủ $K$ tâm cụm.
5. Tiến hành chạy thuật toán Lloyd bình thường với bộ tâm khởi tạo này.

---

## 5. Các Giả định Cấu trúc & Điểm yếu của K-Means

Mặc dù tính toán nhanh và dễ cài đặt, K-Means có những giới hạn toán học khắt khe về dạng dữ liệu:

| Đặc điểm dữ liệu | Ảnh hưởng đến K-Means | Giải pháp thay thế |
|---|---|---|
| Dạng hình học cụm | Chỉ hoạt động tốt với cụm hình cầu/lồi (Convex/Spherical). Thất bại với cụm hình trăng khuyết, xoắn ốc. | DBSCAN, Spectral Clustering |
| Kích thước & Mật độ cụm | Giả định các cụm có mật độ và độ rộng tương đương nhau. Cụm to sẽ nuốt chửng cụm nhỏ. | Gaussian Mixture Models (GMM) |
| Điểm ngoại lệ (Outliers) | Do dùng trung bình cộng ($\mu$), một vài outlier cực đoan sẽ kéo tâm cụm lệch xa thực tế. | K-Medoids / PAM, CLARA |
| Số lượng chiều cao ($d \gg$) | Bị ảnh hưởng bởi "Lời nguyền số chiều" (Curiosity of Dimensionality), khoảng cách Euclidean mất hiệu lực. | PCA / t-SNE giảm chiều trước |

---

## 6. Cài đặt Python Đầy đủ từ Đầu (Full Python Implementation from Scratch, NumPy)

Dưới đây là lớp `KMeansScratch` hoàn chỉnh viết bằng NumPy, tích hợp khởi tạo K-Means++, cơ chế tính toán ma trận tối ưu (Vectorization) và kiểm tra hội tụ.

```python
import numpy as np


class KMeansScratch:

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init="kmeans++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _init_kmeans_plus_plus(self, X):
        """Khởi tạo tâm cụm theo thuật toán K-Means++."""
        n_samples, _ = X.shape
        centroids = np.empty((self.n_clusters, X.shape[1]))

        # Bước 1: Chọn tâm đầu tiên ngẫu nhiên
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]

        # Bước 2-4: Chọn các tâm tiếp theo dựa trên xác suất D(x)^2
        for k in range(1, self.n_clusters):
            # Tính bình phương khoảng cách từ mỗi điểm tới tâm gần nhất
            dist_sq = np.min(
                [np.sum((X - centroids[j]) ** 2, axis=1) for j in range(k)], axis=0
            )

            # Tính phân phối xác suất P(x)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            # Chọn điểm tiếp theo dựa trên xác suất tích lũy
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[k] = X[next_idx]

        return centroids

    def fit(self, X):
        """Huấn luyện K-Means bằng thuật toán Lloyd."""
        n_samples, n_features = X.shape

        # 1. Khởi tạo tâm cụm
        if self.init == "kmeans++":
            self.centroids = self._init_kmeans_plus_plus(X)
        else:
            random_indices = np.random.choice(
                n_samples, self.n_clusters, replace=False
            )
            self.centroids = X[random_indices]

        for iteration in range(self.max_iter):
            # 2. Assignment Step (Tính ma trận khoảng cách Vectorized)
            # Shape: (N, 1, D) - (1, K, D) -> Khoảng cách Euclidean (N, K)
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2
            )
            self.labels = np.argmin(distances, axis=1)

            # 3. Update Step (Tính lại các tâm cụm)
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_mask = self.labels == k
                if np.any(cluster_mask):
                    new_centroids[k] = X[cluster_mask].mean(axis=0)
                else:
                    # Xử lý cụm rỗng: Giữ nguyên tâm cũ
                    new_centroids[k] = self.centroids[k]

            # 4. Kiểm tra điều kiện hội tụ (Convergence Check)
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        # Tính WCSS (Inertia) cuối cùng
        self.inertia_ = np.sum(
            (X - self.centroids[self.labels]) ** 2
        )
        return self


# ==========================================
# CHẠY THỬ THỰC TẾ
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # 1. Sinh dữ liệu giả lập 3 cụm hình cầu
    X, _ = make_blobs(
        n_samples=500, centers=3, cluster_std=0.6, random_state=42
    )

    # 2. Chạy KMeans từ đầu
    model = KMeansScratch(n_clusters=3, init="kmeans++")
    model.fit(X)

    print("=== KẾT QUẢ PHÂN CỤM K-MEANS (FROM SCRATCH) ===")
    print(f"Số vòng lặp thực hiện: {model.max_iter}")
    print(f"Giá trị WCSS (Inertia): {model.inertia_:.4f}")
    print("Vị trí các tâm cụm tìm được:\n", model.centroids)
```
