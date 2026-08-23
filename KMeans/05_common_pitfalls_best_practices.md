---
topic: "K-Means Clustering"
subtopic: "Common Pitfalls, Limitations & Best Practices"
level: "Intermediate"
doc_id: "km_05"
source_url: "https://scikit-learn.org/stable/modules/clustering.html#k-means"
key_concepts:
  - "Giả định cụm hình cầu (Spherical Cluster Assumption)"
  - "Cụm phi lồi & Dạng hình học phức tạp (Non-Convex Clusters)"
  - "Kích thước & Mật độ cụm không đều (Unequal Size & Density)"
  - "Cực tiểu cục bộ & Tham số n_init (Local Minima & n_init)"
  - "Danh mục kiểm định Triển khai (Production Readiness Checklist)"
---

# Những Sai Lầm Thường Gặp & Thực Hành Tốt Nhất (Common Pitfalls & Best Practices)

## 1. Giả định Cụm Hình cầu & Hạn chế Cấu trúc (Spherical Cluster Assumption)

K-Means sử dụng khoảng cách Euclidean để tính WCSS, làm cho ranh giới phân tách giữa các cụm luôn là những đường trung trực hoặc siêu phẳng tuyến tính. Điều này vô hình trung tạo ra giả định rằng tất cả các cụm đều có **dạng hình cầu (Spherical)** và phương sai đẳng hướng.

### Các dạng cấu trúc dữ liệu khiến K-Means thất bại:

* **Cụm Phi lồi (Non-Convex Shapes):** Dữ liệu dạng hình trăng khuyết (`make_moons`), đường xoắn ốc hoặc hình quạt.
* **Cụm Vòng tròn lồng nhau (Concentric Circles):** Cụm bên trong nằm trọn trong cụm bên ngoài (`make_circles`). K-Means sẽ cắt đôi cả hai vòng tròn thay vì phân biệt theo bán kính.
* **Cụm Thuôn dài (Anisotropic / Elongated Clusters):** Cụm có phương sai lệch mạnh về một trục.

```text
[Vòng tròn lồng nhau]       [K-Means Phân chia Sai]
      @@@@@@@                   @@@@@ | @@@@@
    @@  ***  @@               @@  *** | *  @@
    @@  ***  @@      --->     @@  *** | *  @@
      @@@@@@@                   @@@@@ | @@@@@
 (Cụm trong & ngoài)       (Cắt đôi theo trục đứng)
```

---

## 2. Kích thước & Mật độ Cụm Không đều (Unequal Cluster Sizes & Densities)

Cơ chế giảm thiểu WCSS của K-Means tự động ưu tiên các cụm có quy mô phương sai tương đương nhau.

* **Cụm Lớn vs Cụm Nhỏ:** Nếu tập dữ liệu gồm $1$ cụm rất lớn ($1000$ mẫu) và $1$ cụm rất nhỏ ($50$ mẫu), K-Means sẽ cắt nhỏ cụm lớn ra làm đôi để giảm tổng bình phương khoảng cách, đồng thời nuốt chửng cụm nhỏ vào một phần của cụm lớn.
* **Mật độ không đều (Unequal Density):** Các vùng dữ liệu thưa thớt (low-density) nằm cạnh vùng cô đặc (high-density) sẽ bị kéo tâm cụm lệch về phía vùng cô đặc.

---

## 3. Bẫy Cực tiểu Cục bộ & Tham số n_init (Local Minima)

Thuật toán Lloyd là một quy trình tối ưu hóa tham lam (Greedy Algorithm). Nó đảm bảo luôn hội tụ, nhưng chỉ hội tụ về cực tiểu cục bộ (Local Minimum) phụ thuộc hoàn toàn vào vị trí các tâm cụm khởi tạo ban đầu.

```text
Khởi tạo ngẫu nhiên xấu  ---> Lặp Lloyd ---> Cực tiểu cục bộ (Inertia cao)
Khởi tạo K-Means++ tốt  ---> Lặp Lloyd ---> Cực tiểu toàn cục (Inertia thấp)
```

### Chiến lược Khắc phục:

* **Sử dụng `init='k-means++'`:** Giúp trải đều các tâm ban đầu, giảm nguy cơ rơi vào bẫy cực tiểu xấu.
* **Tăng `n_init` (Khuyên dùng $n\_init \ge 10$):** Scikit-Learn sẽ chạy thuật toán $10$ lần độc lập với các hạt giống khác nhau, sau đó chọn ra mô hình duy nhất có `inertia_` thấp nhất.
* **Cố định `random_state`:** Bắt buộc thiết lập trong môi trường Production để đảm bảo kết quả phân cụm có thể tái lập (Reproducible).

---

## 4. Bảng So sánh Thuật toán Phân cụm (Algorithmic Selection)

Khi K-Means gặp các hạn chế cấu trúc trên, ta cần linh hoạt chuyển đổi sang các thuật toán phù hợp hơn:

| Thuật toán | Hình dạng cụm xử lý tốt | Nhạy cảm với Outlier | Độ phức tạp tính toán | Trường hợp ứng dụng ưu tiên |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means** | Hình cầu / Lồi (Spherical / Convex) | Rất cao | $O(N \cdot K \cdot I \cdot d)$ (Rất nhanh) | Dữ liệu quy mô lớn, cụm hình cầu rõ ràng. |
| **DBSCAN** | Mọi hình dạng phức tạp (Phi lồi) | Rất thấp (Tự lọc Noise) | $O(N \log N)$ | Dữ liệu có hình dạng bất quy tắc, nhiều nhiễu. |
| **Gaussian Mixture (GMM)** | Hình Ellipse / Phương sai lệch | Trung bình | $O(N \cdot K \cdot I \cdot d^3)$ | Phân cụm mềm (Soft Clustering), cụm dạng Ellipse. |
| **Agglomerative** | Tùy thuộc vào Linkage | Cao | $O(N^3)$ hoặc $O(N^2 \log N)$ | Dữ liệu quy mô vừa/nhỏ, cần cấu trúc phân cấp (Dendrogram). |

---

## 5. Minh Họa: K-Means Thất Bại & Giải Pháp DBSCAN (Demonstration: K-Means Failure & DBSCAN Solution)

Đoạn mã dưới đây thực nghiệm điểm yếu của K-Means trên tập dữ liệu hình trăng khuyết (`make_moons`) và cách DBSCAN giải quyết triệt để.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# 1. Sinh dữ liệu phi lồi dạng hình trăng khuyết
X, y_true = make_moons(n_samples=500, noise=0.07, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 2. Huấn luyện K-Means (Thất bại do ranh giới tuyến tính)
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3. Huấn luyện DBSCAN (Thành công nhờ phân cụm theo mật độ)
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 4. Đánh giá bằng Adjusted Rand Index (ARI: 1.0 là phân cụm hoàn hảo)
ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
ari_dbscan = adjusted_rand_score(y_true, dbscan_labels)

print("=== BÁO CÁO KẾT QUẢ PHÂN CỤM DỮ LIỆU PHI LỒI ===")
print(f"K-Means ARI Score : {ari_kmeans:.4f} (Thất bại trong việc phân tách)")
print(f"DBSCAN  ARI Score : {ari_dbscan:.4f} (Phân tách hoàn hảo 100%)")

# 5. Trực quan hóa so sánh
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(
    X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap="viridis", edgecolors="k"
)
ax1.set_title(f"K-Means (ARI = {ari_kmeans:.2f}) - Cắt đôi sai cấu trúc")

ax2.scatter(
    X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap="viridis", edgecolors="k"
)
ax2.set_title(f"DBSCAN (ARI = {ari_dbscan:.2f}) - Gom đúng hình dạng")

plt.tight_layout()
plt.show()
```

---

## 6. Production Readiness Checklist (Danh mục Kiểm định Triển khai)

Nghiệm thu mô hình K-Means trước khi đưa vào hệ thống Production:

- [ ] **Xử lý Điểm khuyết & Outliers:** Xác nhận đã xử lý Missing Values và lọc bớt các điểm Outliers cực đoan bằng Z-score/IQR để tránh kéo lệch tâm cụm.
- [ ] **Bắt buộc Chuẩn hóa Thang đo:** Đã đóng gói `StandardScaler` hoặc `RobustScaler` trong Pipeline trước bước KMeans.
- [ ] **Kiểm tra Giả định Cấu trúc:** Xác nhận dữ liệu không có dạng hình học phi lồi (hình trăng khuyết, vòng tròn lồng nhau). Nếu có, chuyển sang DBSCAN.
- [ ] **Biện luận chọn $K$ đa chỉ số:** Đã phối hợp cả đường cong Elbow (Inertia), Silhouette Score và kiến thức chuyên ngành (Domain Knowledge) để chốt $K$.
- [ ] **Cấu hình Cực tiểu Cục bộ:** Cấu hình `init='k-means++'` và `n_init >= 10`.
- [ ] **Tái lập Kết quả (Reproducibility):** Đã cố định tham số `random_state`.
- [ ] **Trích xuất Pipeline:** Đóng gói mô hình dưới dạng `sklearn.pipeline.Pipeline` để tiện áp dụng `.predict()` hoặc `.transform()` cho dữ liệu mới.
