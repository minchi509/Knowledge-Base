---
topic: "K-Means Clustering"
subtopic: "Scikit-Learn API & Hyperparameters"
level: "Intermediate"
doc_id: "km_02"
source_url: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"
key_concepts:
  - "Số lượng cụm (n_clusters)"
  - "Khởi tạo tâm cụm (init & n_init)"
  - "Điều kiện hội tụ (max_iter & tol)"
  - "Biến đổi khoảng cách (transform)"
  - "Biến thể thuật toán Elkan vs Lloyd (algorithm)"
---

# Scikit-Learn API & Hyperparameter Matrix

## 1. Class Specification (`sklearn.cluster.KMeans`)

Lớp `sklearn.cluster.KMeans` cung cấp giao diện chuẩn để thực thi phân cụm K-Means.

### 1.1. Các phương thức cốt lõi (Core Methods)
* **`.fit(X)`**: Tìm vị trí các tâm cụm tối ưu từ tập dữ liệu $X$.
* **`.fit_predict(X)`**: Thực hiện `.fit(X)` và trả về ngay mảng nhãn cụm (Cluster Labels) tương ứng cho từng mẫu trong $X$.
* **`.predict(X)`**: Dự đoán cụm cho các mẫu dữ liệu *mới* bằng cách gán chúng vào tâm cụm gần nhất đã học được từ bước `fit`.
* **`.transform(X)`**: Biến đổi $X$ từ không gian $d$-chiều ban đầu sang không gian $K$-chiều. Giá trị tại cột $j$ đại diện cho **khoảng cách Euclidean** từ mẫu đó tới tâm cụm $\mu_j$. *(Rất hữu ích cho Feature Engineering)*.

### 1.2. Thuộc tính quan trọng sau khi Fit (Key Attributes)
* **`.cluster_centers_`**: Mảng NumPy hình dạng `(n_clusters, n_features)` chứa tọa độ các tâm cụm sau khi hội tụ.
* **`.labels_`**: Mảng hình dạng `(n_samples,)` chứa chỉ số cụm ($0$ đến $K-1$) của từng điểm dữ liệu tập Train.
* **`.inertia_`**: Giá trị WCSS (Within-Cluster Sum of Squares) tối ưu đạt được tại vòng lặp cuối cùng.
* **`.n_iter_`**: Số lượng vòng lặp thực tế mà thuật toán đã thực hiện trước khi hội tụ.

---

## 2. Hyperparameter Configuration Matrix

| Tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa trực quan & Hướng dẫn cấu hình |
| :--- | :--- | :--- | :--- |
| `n_clusters` | `int` | `8` | **Số cụm $K$ cần phân chia.** Đây là tham số quan trọng nhất. Cần kết hợp Phương pháp Khuỷu tay (Elbow) và Chỉ số Silhouette để chọn. |
| `init` | `str` / `array` | `'k-means++'` | Phương pháp chọn tâm ban đầu: `'k-means++'`, `'random'`, hoặc mảng tọa độ truyền vào. **Luôn giữ mặc định `'k-means++'`**. |
| `n_init` | `int` / `str` | `'auto'` | Số lần thuật toán chạy độc lập với các hạt giống (seeds) ngẫu nhiên khác nhau. Kết quả có `inertia_` thấp nhất sẽ được chọn. Khuyên dùng $10$. |
| `max_iter` | `int` | `300` | Số lượng vòng lặp tối đa trong một lần chạy. Nếu mô hình chưa kịp hội tụ, cân nhắc tăng lên $500$. |
| `tol` | `float` | `1e-4` | Ngưỡng dung sai (Tolerance). Nếu tổng sự dịch chuyển vị trí các tâm cụm nhỏ hơn `tol`, thuật toán dừng lặp. |
| `algorithm` | `str` | `'lloyd'` | Biến thể thuật toán: `'lloyd'` (chuẩn) hoặc `'elkan'`. |

### Phân biệt Biến thể Thuật toán: `'lloyd'` vs `'elkan'`
* **`algorithm='lloyd'`**: Thuật toán K-Means truyền thống. Phù hợp với mọi dạng dữ liệu tổng quát nhưng phải tính lại khoảng cách giữa tất cả các điểm và tất cả các tâm ở mỗi vòng lặp.
* **`algorithm='elkan'`**: Tận dụng **Bất đẳng thức tam giác (Triangle Inequality)** để bỏ qua các phép tính khoảng cách không cần thiết giữa điểm dữ liệu và các tâm cụm ở xa. Chạy nhanh hơn đáng kể trên dữ liệu có cụm rõ ràng, nhưng tốn nhiều bộ nhớ RAM hơn ($O(N \cdot K)$).

---

## 3. Lưu ý Thực chiến & Kỹ thuật Feature Engineering

### 3.1. Bắt buộc phải Chuẩn hóa Dữ liệu (Scaling)
Do K-Means đo đạc độ tương đồng dựa hoàn toàn vào khoảng cách Euclidean $\Vert x_i - \mu_j \Vert$, nếu một đặc trưng có biên độ lớn (ví dụ `Thu_Nhap`: $10,000,000 - 100,000,000$) và đặc trưng khác có biên độ nhỏ (ví dụ `Tuoi`: $18 - 65$), đặc trưng lớn sẽ áp đảo hoàn toàn khoảng cách. 

-> **Cần luôn kết hợp `StandardScaler` trong `Pipeline` trước khi gọi `KMeans`.**

### 3.2. Biến đổi Không gian Khoảng cách với `.transform()`
Thay vì chỉ dùng chỉ số cụm (dạng Categorical $0, 1, 2$), ta có thể dùng `.transform(X)` để biến đổi một mẫu $x \in \mathbb{R}^d$ thành vector khoảng cách $x' \in \mathbb{R}^K$. Các giá trị khoảng cách liên tục này có thể làm đầu vào (non-linear features) rất tốt cho các mô hình học có giám sát như Linear Regression hay Logistic Regression.

---

## 4. Complete Production Executable Pipeline

Đoạn mã hoàn chỉnh dưới đây đóng gói toàn bộ quy trình tiền xử lý, phân cụm K-Means, trích xuất chỉ số và đưa vào dự đoán thực tế.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Tạo tập dữ liệu giả lập 4 cụm trong không gian 3 chiều
X, y_true = make_blobs(
    n_samples=1000,
    n_features=3,
    centers=4,
    cluster_std=0.9,
    random_state=42
)

# 2. Xây dựng Production Pipeline
# Chuẩn hóa dữ liệu bắt buộc bằng StandardScaler trước khi vào KMeans
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=4,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        algorithm='lloyd',
        random_state=42
    ))
])

# 3. Huấn luyện và dự đoán nhãn cụm cho tập dữ liệu Train
cluster_labels = pipeline.fit_predict(X)

# Trích xuất đối tượng KMeans đã huấn luyện từ Pipeline
kmeans_model = pipeline.named_steps['kmeans']
scaler_model = pipeline.named_steps['scaler']

print("=== BÁO CÁO KẾT QUẢ PHÂN CỤM K-MEANS ===")
print(f"Tổng giá trị WCSS (Inertia): {kmeans_model.inertia_:.2f}")
print(f"Số vòng lặp thực tế để hội tụ: {kmeans_model.n_iter_}")
print("\nTọa độ các Centroids (trên không gian đã chuẩn hóa Standardized):")
print(kmeans_model.cluster_centers_)

# Quy đổi tọa độ Centroids về lại thang đo gốc ban đầu (Original Scale)
original_centroids = scaler_model.inverse_transform(kmeans_model.cluster_centers_)
print("\nTọa độ các Centroids (trên thang đo ban đầu):")
print(original_centroids)

# 4. Dự đoán và Biến đổi khoảng cách trên Dữ liệu Mới
X_new = np.array([
    [1.5, -2.0, 3.1],
    [-5.0, 8.2, -1.1]
])

# Dự đoán cụm cho mẫu mới
new_labels = pipeline.predict(X_new)

# Trích xuất ma trận khoảng cách đến 4 tâm cụm
# Cần scale dữ liệu mới trước khi transform qua kmeans
X_new_scaled = scaler_model.transform(X_new)
distances_to_centroids = kmeans_model.transform(X_new_scaled)

print("\n=== DỰ ĐOÁN DỮ LIỆU MỚI ===")
for i, sample in enumerate(X_new):
    print(f"Mẫu {i+1} {sample} -> Thuộc Cụm {new_labels[i]}")
    print(f"   Khoảng cách đến 4 tâm cụm: {distances_to_centroids[i].round(3)}\n")
