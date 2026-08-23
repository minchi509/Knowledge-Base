---
topic: "K-Means Clustering"
subtopic: "Data Preprocessing, Feature Scaling & Dimensionality Reduction"
level: "Intermediate"
doc_id: "km_03"
sources:
  - "Scikit-Learn User Guide: Preprocessing"
key_concepts:
  - "Bóp méo khoảng cách Euclidean (Euclidean Distance Distortion)"
  - "Chuẩn hóa dữ liệu với StandardScaler (StandardScaler)"
  - "Độ nhạy với điểm ngoại lệ (Outlier Sensitivity)"
  - "Tích hợp PCA giảm số chiều (PCA Integration)"
---

# Hướng Dẫn Tiền Xử Lý & Giảm Số Chiều (Preprocessing & Dimensionality Reduction Guidelines)

## 1. Mức độ nhạy cảm với Thang đo (Scaling Sensitivity)

K-Means tính toán độ tương đồng giữa các điểm dựa hoàn toàn vào khoảng cách Euclidean trong không gian $d$-chiều:

$$d(x_i, x_j) = \sqrt{\sum_{m=1}^{d} (x_{im} - x_{jm})^2}$$

### Hậu quả của hiện tượng Bóp méo Khoảng cách (Euclidean Distance Distortion)

Khi các đặc trưng (features) có đơn vị đo hoặc biên độ (scale) khác biệt lớn, đặc trưng có khoảng giá trị rộng hơn sẽ chiếm ưu thế tuyệt đối trong công thức tính khoảng cách, khiến các đặc trưng còn lại bị triệt tiêu hoàn toàn ảnh hưởng.

* **Ví dụ thực tế:** Xét bộ dữ liệu phân khúc khách hàng gồm 2 đặc trưng:
  * `Tuoi` (Tuổi): $18 \to 70$ (độ lệch chênh lệch tối đa $\Delta \approx 50$).
  * `Thu_Nhap` (Thu nhập năm): $15,000 \to 150,000$ USD (độ lệch $\Delta \approx 135,000$).
* **Ảnh hưởng toán học:** Khi tính bình phương khoảng cách $(x_{i1} - x_{j1})^2 + (x_{i2} - x_{j2})^2$, biến `Thu_Nhap` đóng góp đến $99.99\%$ giá trị khoảng cách, biến `Tuoi` hoàn toàn không có trọng số trong việc định hình tâm cụm.

### Giải pháp Chuẩn hóa

Luôn áp dụng **StandardScaler** (Z-score Normalization) để đưa tất cả đặc trưng về cùng phân phối có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$:

$$z = \frac{x - \mu}{\sigma}$$

| Phương pháp Scaler | Công thức | Trường hợp sử dụng ưu tiên cho K-Means |
| :--- | :--- | :--- |
| **StandardScaler** | $z = \frac{x - \mu}{\sigma}$ | Mặc định cho K-Means (giữ nguyên hình dạng phân phối chuẩn). |
| **MinMaxScaler** | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Phù hợp khi muốn đưa dữ liệu về khoảng cố định $[0, 1]$. |
| **RobustScaler** | $x_{\text{scaled}} = \frac{x - Q_2}{Q_3 - Q_1}$ | Tốt hơn khi dữ liệu chứa nhiều điểm ngoại lệ (Outliers). |

---

## 2. Độ nhạy với Điểm ngoại lệ (Outlier Sensitivity)

Hàm mục tiêu WCSS của K-Means tính bình phương khoảng cách Euclidean $\|x_i - \mu_j\|^2$. Do phép nâng lên bình phương, các điểm dữ liệu nằm xa quần thể (Outliers) sẽ đóng góp mức độ sai số cực lớn.

```text
[ Cụm chính (Cluster) ] -----------------------------> (Outlier cực xa)
                                  ^
                                  |
                Tâm cụm bị kéo lệch mạnh về phía Outlier
```

**Cơ chế ảnh hưởng:** Thuật toán Lloyd bị buộc phải kéo tâm cụm $\mu_j$ về phía các outlier để giảm thiểu giá trị bình phương lỗi của điểm ngoại lệ đó, khiến tâm cụm không còn phản ánh đúng trung tâm của đa số dữ liệu.

**Chiến lược xử lý:**

1. **Lọc Outlier trước khi gom cụm:** Sử dụng IQR (Interquartile Range) hoặc Z-Score để gạch bỏ các điểm ngoại lệ trước khi đưa vào KMeans.
2. **Thay đổi thuật toán:** Chuyển sang K-Medoids (PAM) (dùng điểm trung vị thay vì trung bình) hoặc DBSCAN (tự động gom nhiễu vào nhóm Noise) nếu không thể xóa bỏ outlier.

---

## 3. Tích hợp PCA xử lý Lời nguyền số chiều (Curse of Dimensionality)

Khi số lượng đặc trưng tăng lên lớn ($d > 10$), K-Means gặp phải hiện tượng Lời nguyền số chiều (Curse of Dimensionality):

* Không gian trở nên cực kỳ thưa thớt (sparse).
* Khoảng cách Euclidean giữa bất kỳ cặp điểm nào cũng đều tiến về một giá trị tương đương ($d_{\max} \approx d_{\min}$), khiến khái niệm "gần - xa" mất ý nghĩa.
* Dữ liệu chứa nhiều nhiễu và các thuộc tính đa cộng tuyến (multicollinearity).

### Kỹ thuật Tích hợp PCA (Principal Component Analysis)

Giải pháp chuẩn là áp dụng PCA để giảm không gian $d$-chiều xuống $k$-chiều ($k \ll d$) chứa các thành phần chính thể hiện phần lớn phương sai của dữ liệu trước khi đưa vào K-Means.

```text
[Data Gốc (High-D)] ---> [StandardScaler] ---> [PCA (Giữ 90-95% Variance)] ---> [K-Means]
```

---

## 4. Cài đặt Pipeline Hoàn chỉnh cho Production (Complete Production Pipeline Implementation)

Đoạn mã dưới đây minh họa toàn bộ quy trình: Tạo dữ liệu đa chiều kèm nhiễu/outliers, đóng gói Pipeline gồm StandardScaler, PCA (giữ $90\%$ phương sai) và KMeans, sau đó đánh giá kết quả.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Tạo tập dữ liệu giả lập 20 chiều với 4 cụm thực tế
X_raw, y_true = make_blobs(
    n_samples=800,
    n_features=20,
    centers=4,
    cluster_std=2.5,
    random_state=42,
)

# Thêm 20 điểm Outlier cực đoan vào dữ liệu
np.random.seed(42)
outliers = np.random.uniform(low=-50, high=50, size=(20, 20))
X_data = np.vstack([X_raw, outliers])

# 2. Xây dựng Production Pipeline hoàn chỉnh
# Bước 1: StandardScaler chuẩn hóa thang đo các thuộc tính
# Bước 2: PCA giảm số chiều, giữ lại 90% tổng phương sai
# Bước 3: KMeans thực hiện phân cụm trên không gian mới
clustering_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90, random_state=42)),
    ("kmeans", KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)),
])

# 3. Huấn luyện Pipeline
cluster_labels = clustering_pipeline.fit_predict(X_data)

# 4. Trích xuất chỉ số kiểm tra từng công đoạn
scaler_step = clustering_pipeline.named_steps["scaler"]
pca_step = clustering_pipeline.named_steps["pca"]
kmeans_step = clustering_pipeline.named_steps["kmeans"]

print("=== BÁO CÁO TIỀN XỬ LÝ & GIẢM CHIỀU ===")
print(f"Số chiều ban đầu: {X_data.shape[1]} chiều")
print(f"Số chiều sau PCA (giữ 90% variance): {pca_step.n_components_} chiều")
print(
    f"Tổng phương sai giải thích được (Explained Variance Ratio): {np.sum(pca_step.explained_variance_ratio_):.4f}"
)

# 5. Đánh giá chất lượng phân cụm bằng Silhouette Score
X_transformed = pca_step.transform(scaler_step.transform(X_data))
sil_score = silhouette_score(X_transformed, cluster_labels)

print("\n=== ĐÁNH GIÁ MÔ HÌNH K-MEANS ===")
print(f"Giá trị WCSS (Inertia): {kmeans_step.inertia_:.2f}")
print(f"Silhouette Score: {sil_score:.4f}")
```
