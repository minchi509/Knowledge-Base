---
topic: "K-Means Clustering"
subtopic: "Common Pitfalls, Limitations & Best Practices"
level: "Intermediate"
doc_id: "km_05"
sources:
  - "Scikit-Learn Clustering Limitations"
key_concepts:
  - "Spherical Cluster Assumption"
  - "Non-Convex Clusters"
  - "Unequal Cluster Sizes"
  - "Production Readiness Checklist"
---

# Common Pitfalls & Best Practices

## 1. Giả định Cụm Hình cầu (Spherical Cluster Assumption)
**Cạm bẫy**: K-Means ngầm định rằng các cụm có dạng **hình cầu (spherical)** và phương sai đẳng hướng.
Thuật toán sẽ thất bại hoàn toàn nếu các cụm có hình dạng phức tạp:
* Hình bán nguyệt/mặt trăng (Moons).
* Các vòng tròn lồng nhau (Concentric circles).
* Cụm hình thuôn dài có mật độ khác nhau.

**Khắc phục**: Khi dữ liệu có hình dạng bất quy tắc, chuyển sang dùng `DBSCAN` hoặc `AgglomerativeClustering`.

## 2. Kích thước và Mật độ cụm không đều (Unequal Density/Size)
K-Means có xu hướng chia dữ liệu thành các cụm có kích thước bằng nhau. Nếu tập dữ liệu gồm 1 cụm rất lớn (900 mẫu) và 1 cụm rất nhỏ (100 mẫu), K-Means sẽ có xu hướng cắt cụm lớn ra làm đôi để đưa về quy mô tương đương.

## 3. Nhạy cảm với giá trị khởi tạo
Mặc dù `k-means++` giảm thiểu rủi ro, K-Means vẫn là thuật toán tối ưu cục bộ.
**Khắc phục**: Đảm bảo tham số `n_init` $\ge 10$ trong môi trường thực tế để chọn ra kết quả tốt nhất từ 10 lần khởi tạo ngẫu nhiên khác nhau.

## 4. Production Readiness Checklist
Trước khi nghiệm thu Notebook/Script phân cụm bằng K-Means:
- [ ] Bắt buộc đã xử lý giá trị khuyết (`Missing values`) và Outliers.
- [ ] Đã áp dụng `StandardScaler` trước khi đưa vào K-Means.
- [ ] Đã biện luận chọn $K$ dựa trên sự kết hợp của cả **Elbow Plot** và **Silhouette Score**.
- [ ] Cố định `random_state` để kết quả phân cụm có thể tái lập (Reproducible).
