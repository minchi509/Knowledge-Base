---
topic: "K-Means Clustering"
subtopic: "Data Preprocessing, Scaling & Outliers"
level: "Intermediate"
doc_id: "km_03"
sources:
  - "Scikit-Learn User Guide: Preprocessing"
key_concepts:
  - "Distance Metric Sensitivity"
  - "StandardScaler"
  - "Outlier Impact"
---

# Preprocessing Guidelines for K-Means

## 1. Mức độ nhạy cảm với Thang đo (Scale Sensitivity)
**Nguyên tắc vàng:** K-Means bắt buộc phải được chuẩn hóa dữ liệu trước khi huấn luyện.
*Giải thích*: K-Means sử dụng khoảng cách Euclidean. Giả sử tập dữ liệu *Mall Customer* có cột `Tuổi` (phạm vi từ $18$ đến $70$) và `Thu nhập hàng năm` (phạm vi từ $15,000$ đến $137,000$ USD). Biến số `Thu nhập` có độ lớn áp đảo hoàn toàn biến `Tuổi`. Khi tính khoảng cách, thuật toán gần như chỉ quan tâm đến sự chênh lệch thu nhập mà phớt lờ độ tuổi.

**Giải pháp:** Áp dụng `StandardScaler` (hoặc `MinMaxScaler`) để đưa tất cả các biến về cùng một phương sai và trung bình.

## 2. Xử lý Nhiễu (Outliers)
K-Means vô cùng nhạy cảm với Outliers. Vì thuật toán tối ưu hóa "bình phương khoảng cách" (Sum of Squares), một điểm nhiễu nằm quá xa sẽ kéo lệch vị trí của tâm cụm (centroid) về phía nó, làm méo mó hình dạng của toàn bộ cụm.
**Giải pháp trong quá trình EDA:**
* Sử dụng Boxplot để phát hiện Outliers.
* Loại bỏ (Drop) Outliers trước khi đưa vào mô hình, hoặc đổi sang thuật toán K-Medoids (sử dụng trung vị thay vì trung bình).

## 3. Production Pipeline for Clustering
Đối với biến phân loại (như `Gender` trong Mall Customer), K-Means chuẩn không xử lý tốt (do khoảng cách giữa 'Male' và 'Female' là vô nghĩa trên không gian liên tục). Ở mức Intermediate, ta thường mã hóa nhị phân (0-1) cho biến có 2 giá trị, hoặc chỉ dùng các biến số thực (`Age`, `Annual Income`, `Spending Score`) để gom cụm.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Xây dựng Pipeline chuẩn mực (Vừa Scale vừa Cluster)
clustering_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=5, init='k-means++', random_state=42))
])

# Dữ liệu khách hàng
X_mall = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Chạy pipeline
cluster_labels = clustering_pipeline.fit_predict(X_mall)
