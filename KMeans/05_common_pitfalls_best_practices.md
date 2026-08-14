---
topic: "K-Means Clustering"
subtopic: "Common Pitfalls & Best Practices"
level: "Intermediate"
doc_id: "km_05"
sources:
  - "Scikit-Learn Common Pitfalls: Clustering"
key_concepts:
  - "Spherical Assumption"
  - "Curse of Dimensionality"
  - "Production Checklist"
---

# Common Pitfalls & Best Practices

## 1. Giả định Hình học của K-Means (Spherical Assumption)
**Pitfall lớn nhất:** K-Means ngầm định rằng các cụm có hình cầu (spherical) và có phương sai tương đồng nhau (kích thước cụm bằng nhau). 
Nếu dữ liệu thực tế phân bố theo hình thuôn dài (elliptical), hình mặt trăng (moons), hay các vòng tròn lồng vào nhau, K-Means sẽ chia cắt dữ liệu cực kỳ phi lý. 
**Giải pháp:** Trong những trường hợp dữ liệu phi chuẩn, cần chuyển sang dùng `DBSCAN` (gom cụm dựa trên mật độ) hoặc `GaussianMixture` (cho phép cụm hình elip).

## 2. Lời nguyền Chiều dữ liệu (Curse of Dimensionality)
Bởi vì K-Means sử dụng khoảng cách Euclidean, khi số lượng đặc trưng (số cột) quá lớn ($> 50$), khoảng cách giữa mọi cặp điểm đều tiến về gần xấp xỉ nhau. Điều này khiến khái niệm "gần" hay "xa" mất đi ý nghĩa toán học.
**Giải pháp:** Nếu dữ liệu có quá nhiều chiều, bắt buộc phải áp dụng Giảm chiều dữ liệu (ví dụ: `PCA - Principal Component Analysis`) trước khi đưa vào `KMeans`.

## 3. Production Readiness Checklist
Trước khi đưa Notebook K-Means vào đánh giá và sinh bài tập:
- [ ] Dữ liệu MỚI CHỈ CÓ TÍNH CHẤT SỐ (Numeric) đã được lọc ra để đưa vào thuật toán chưa?
- [ ] Bắt buộc phải có bước sử dụng `StandardScaler` (Nếu thiếu, Agent cần đánh trượt Notebook).
- [ ] Bắt buộc phải có Code Block minh họa vòng lặp For để vẽ biểu đồ Elbow Method.
- [ ] Tham số `random_state` phải được cố định để giám khảo chấm bài (Auto-Grader) kiểm tra được sự trùng khớp của mảng `cluster_labels`.
- [ ] Đã in ra được Silhouette Score để đánh giá định lượng chất lượng cụm chưa?
