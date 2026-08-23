---
topic: "K-Means Clustering"
subtopic: "Optimal K Selection & Diagnostic Metrics"
level: "Intermediate"
doc_id: "km_04"
source_url: "https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation"
key_concepts:
  - "Phương pháp Khuỷu tay (Elbow Method)"
  - "Hệ số Silhouette (Silhouette Coefficient)"
  - "Chỉ số Davies-Bouldin (Davies-Bouldin Index)"
  - "Trực quan hóa & Đánh giá Cụm (Cluster Visualization & Diagnostics)"
---

# Chẩn Đoán & Lựa Chọn K Tối Ưu (Diagnostics & Optimal K Selection)

Do K-Means là thuật toán Học không giám sát (Unsupervised Learning), tập dữ liệu không có nhãn thực tế (Ground-Truth Labels) để tính toán các chỉ số như Accuracy, Precision hay Recall. Việc xác định số cụm tối ưu $K$ và đánh giá chất lượng phân cụm dựa hoàn toàn vào các **chỉ số cấu trúc hình học (Intrinsic Geometry Metrics)**.

---

## 1. Phương pháp Khuỷu tay (Elbow Method)

Phương pháp Khuỷu tay theo dõi sự thay đổi của tổng bình phương khoảng cách nội cụm **WCSS (Inertia)** theo số lượng cụm $K$.

### 1.1. Nguyên lý Hoạt động

* Khi $K$ tăng, các tâm cụm nằm gần các điểm dữ liệu hơn, do đó WCSS $J(K)$ luôn giảm đơn điệu.
* **Vùng suy giảm nhanh ($K < K_{\text{optimal}}$):** Việc bổ sung tâm cụm giúp gom các điểm ở xa vào đúng cụm, làm WCSS giảm rất mạnh.
* **Vùng suy giảm chậm ($K > K_{\text{optimal}}$):** Việc chia nhỏ thêm các cụm vốn đã gộp tốt chỉ làm giảm nhẹ WCSS.
* **Điểm Khuỷu tay (Elbow Point):** Điểm uốn trên đồ thị — nơi tốc độ giảm WCSS chuyển từ nhanh sang chậm. Giá trị $K$ tại vị trí này được chọn làm số cụm tối ưu.

```text
Inertia (WCSS)
  ^
  |  *
  |   \
  |    \
  |     * (Elbow Point: K=3)
  |      \__________*_________*
  +------------------------------> Số cụm (K)
```

### 1.2. Hạn chế

* Nhận diện mang tính cảm quan (subjective), khó xác định nếu đường cong giảm đều mượt mà.
* Cần kết hợp thêm các chỉ số định lượng khắt khe hơn như Silhouette hay Davies-Bouldin.

---

## 2. Hệ số Silhouette (Silhouette Coefficient)

Hệ số Silhouette đánh giá chất lượng phân cụm của từng điểm dữ liệu bằng cách cân bằng giữa Độ gắn kết nội cụm (Cohesion) và Độ phân tách ngoại cụm (Separation).

### 2.1. Công thức Toán học

Với một mẫu dữ liệu $x_i$ thuộc cụm $C_A$:

**Độ gắn kết nội cụm $a(i)$:** Khoảng cách trung bình từ $x_i$ đến tất cả các điểm khác trong cùng cụm $C_A$:

$$a(i) = \frac{1}{|C_A| - 1} \sum_{x_j \in C_A, j \neq i} \|x_i - x_j\|$$

**Độ phân tách ngoại cụm $b(i)$:** Khoảng cách trung bình từ $x_i$ đến tất cả các điểm trong cụm gần nhất $C_B$ ($C_B \neq C_A$):

$$b(i) = \min_{C_k \neq C_A} \left( \frac{1}{|C_k|} \sum_{x_j \in C_k} \|x_i - x_j\| \right)$$

**Hệ số Silhouette $s(i)$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### 2.2. Thang Giá trị & Trực quan hóa

Hệ số Silhouette có giá trị nằm trong khoảng $[-1, +1]$:

* $s(i) \approx +1$: Mẫu $x_i$ nằm rất xa cụm lân cận và rất gần các điểm trong cụm (Phân cụm xuất sắc).
* $s(i) \approx 0$: Mẫu $x_i$ nằm ngay ranh giới giữa 2 cụm (Ranh giới mơ hồ).
* $s(i) \approx -1$: Mẫu $x_i$ nằm gần cụm lân cận hơn cụm hiện tại (Gán sai cụm).

**Silhouette Score toàn cục:** Trung bình cộng $s(i)$ của tất cả các mẫu dữ liệu. Giá trị $K$ có Silhouette Score cao nhất là lựa chọn tối ưu.

---

## 3. Chỉ số Davies-Bouldin (Davies-Bouldin Index - DBI)

Chỉ số Davies-Bouldin đo lường độ tương đồng trung bình giữa mỗi cụm và cụm giống nó nhất.

### 3.1. Công thức Toán học

Độ tương đồng $R_{ij}$ giữa cụm $C_i$ và cụm $C_j$:

$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}$$

Trong đó:

* $s_i$: Đường kính trung bình của cụm $C_i$ (độ phân tán nội cụm).
* $d(\mu_i, \mu_j)$: Khoảng cách giữa 2 tâm cụm $\mu_i$ và $\mu_j$ (độ phân tách).

Chỉ số Davies-Bouldin toàn cục:

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} R_{ij}$$

### 3.2. Quy tắc Đánh giá

* Giá trị $DB \ge 0$.
* Chỉ số $DB$ càng nhỏ (tiến về $0$), chất lượng phân cụm càng tốt (cụm càng thu gọn nội bộ và càng cách xa các cụm khác).

---

## 4. Bảng So sánh Tổng hợp các Chỉ số Đánh giá Cụm

| Chỉ số (Metric) | Phạm vi Giá trị | Hướng Tối ưu | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- | :--- |
| **Inertia (WCSS)** | $[0, +\infty)$ | Càng nhỏ càng tốt (tìm điểm Elbow) | Tính toán cực nhanh, trực quan. | Luôn giảm khi $K$ tăng, dễ phụ thuộc cảm quan. |
| **Silhouette Score** | $[-1, +1]$ | Càng LỚN càng tốt | Đánh giá chính xác độ phân tách và thu gọn. | Độ phức tạp tính toán cao $O(N^2)$. |
| **Davies-Bouldin Index** | $[0, +\infty)$ | Càng NHỎ càng tốt | Chẩn đoán nhanh độ chồng lấp giữa các cụm. | Nhạy cảm với các cụm không có dạng hình cầu. |
| **Calinski-Harabasz** | $[0, +\infty)$ | Càng LỚN càng tốt | Tính toán nhanh, hiệu quả với cụm lồi rõ ràng. | Kém chính xác khi các cụm có mật độ khác nhau. |

---

## 5. Kịch bản Đánh giá Chẩn đoán Đầy đủ (Complete Diagnostic Evaluation Script: Elbow + Silhouette + Davies-Bouldin)

Đoạn mã Python thực thi dưới đây tính toán đồng thời 3 chỉ số và vẽ đồ thị đa trục giúp chọn $K$ chính xác.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Sinh dữ liệu giả lập 4 cụm thực tế
X_raw, _ = make_blobs(
    n_samples=600, centers=4, cluster_std=0.8, random_state=42
)
X_scaled = StandardScaler().fit_transform(X_raw)


# 2. Hàm đánh giá chuỗi giá trị K
def evaluate_kmeans_diagnostics(X, k_range=range(2, 11)):
    inertias = []
    silhouette_scores = []
    db_indices = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)

        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        db_indices.append(davies_bouldin_score(X, labels))

    # 3. Trực quan hóa kết quả trên 3 Đồ thị
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Đồ thị 1: Elbow Method (Inertia)
    ax1.plot(k_range, inertias, "o-", color="tab:blue", linewidth=2)
    ax1.set_title("1. Phương pháp Khuỷu tay (Elbow)", fontsize=12)
    ax1.set_xlabel("Số cụm (K)")
    ax1.set_ylabel("WCSS (Inertia)")
    ax1.grid(True)

    # Đồ thị 2: Silhouette Score (Cao nhất = Tốt nhất)
    ax2.plot(k_range, silhouette_scores, "s--", color="tab:green", linewidth=2)
    best_k_sil = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(
        x=best_k_sil,
        color="red",
        linestyle=":",
        label=f"Max Silhouette (K={best_k_sil})",
    )
    ax2.set_title("2. Hệ số Silhouette (Cao = Tốt)", fontsize=12)
    ax2.set_xlabel("Số cụm (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    ax2.grid(True)

    # Đồ thị 3: Davies-Bouldin Index (Thấp nhất = Tốt nhất)
    ax3.plot(k_range, db_indices, "d-.", color="tab:red", linewidth=2)
    best_k_db = k_range[np.argmin(db_indices)]
    ax3.axvline(
        x=best_k_db,
        color="blue",
        linestyle=":",
        label=f"Min Davies-Bouldin (K={best_k_db})",
    )
    ax3.set_title("3. Chỉ số Davies-Bouldin (Thấp = Tốt)", fontsize=12)
    ax3.set_xlabel("Số cụm (K)")
    ax3.set_ylabel("Davies-Bouldin Index")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return best_k_sil, best_k_db


# Chạy hàm đánh giá
best_k_sil, best_k_db = evaluate_kmeans_diagnostics(X_scaled)
print(f"-> K tối ưu theo Silhouette Score: K = {best_k_sil}")
print(f"-> K tối ưu theo Davies-Bouldin Index: K = {best_k_db}")
```
