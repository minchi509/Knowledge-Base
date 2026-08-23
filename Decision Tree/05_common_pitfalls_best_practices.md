---
topic: "Decision Tree"
subtopic: "Common Pitfalls, Overfitting & Cost-Complexity Pruning"
level: "Intermediate"
doc_id: "dt_05"
source_url: "https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning"
key_concepts:
  - "Phương sai cao & Quá khớp (High Variance & Overfitting)"
  - "Điểm mù Ngoại suy (Extrapolation Blindspot)"
  - "Độ nhạy với Phép xoay (Rotational Sensitivity)"
  - "Cắt tỉa Tối ưu Chi phí (Cost-Complexity Pruning - ccp_alpha)"
  - "Công thức Effective Alpha"
  - "Danh mục Kiểm định Triển khai (Production Readiness Checklist)"
---

# Những Sai Lầm Thường Gặp & Thực Hành Cắt Tỉa Cây (Common Pitfalls & Pruning Practices)

## 1. High Variance & Overfitting Syndrome (Phương sai cao & Quá khớp)

Decision Tree là một mô hình **phi tham số (non-parametric)** có độ linh hoạt (capacity) rất lớn. Nếu không được kiểm soát, cây sẽ tiếp tục chia nhánh cho đến khi mọi nút lá đều thuần khiết $100\%$.

### 1.1. Bản chất phương sai cao (High Variance)

* **Độ nhạy dữ liệu (Data Instability):** Do thuật toán CART áp dụng chiến lược **Tham lam (Greedy Choice)** tại từng bước, một sự thay đổi nhỏ trong dữ liệu huấn luyện (ví dụ: thêm hoặc bớt $1\%$ mẫu dữ liệu) có thể làm thay đổi hoàn toàn câu hỏi ở Nút Gốc (Root Node). Điều này dẫn đến toàn bộ cấu trúc cây bên dưới bị biến đổi hoàn toàn.
* **Biểu hiện thực tế:**
  * Độ chính xác trên tập Train: $99\% - 100\%$ (Mô hình nhớ thuộc lòng nhiễu).
  * Độ chính xác trên tập Test: Sụt giảm nghiêm trọng xuống $50\% - 65\%$.

---

## 2. Structural Limitations & Blindspots (Các điểm mù cấu trúc)

Ngoài Overfitting, Cây quyết định còn mắc phải $2$ hạn chế cấu trúc cốt lõi mà người làm dữ liệu bắt buộc phải nắm rõ:

### 2.1. Điểm mù Ngoại suy (Extrapolation Blindspot)

Decision Tree **tuyệt đối không có khả năng ngoại suy (Extrapolate)** ra ngoài khoảng dữ liệu đã thấy trong tập Train.

* **Cơ chế dự đoán:** Cây quyết định chia không gian thành các vùng hình chữ nhật và đưa ra dự đoán dạng **Hằng số theo từng khoảng (Piecewise Constant)**.
* **Hậu quả:** Với bài toán Hồi quy (Regression), nếu giá trị đầu vào $X_{\text{test}}$ nằm ngoài khoảng $[X_{\min}, X_{\max}]$ của tập Train, Cây chỉ đơn giản gán giá trị dự đoán bằng trung bình của nút lá gần nhất, thay vì nắm bắt xu hướng tăng/giảm (Trend) như Linear Regression.

```text
[Linear Regression]  -----> Dự đoán tăng tiến theo xu hướng (Ngoại suy tốt)
[Decision Tree]      -----> Dự đoán đi ngang thành đường thẳng hằng số (Ngoại suy thất bại)
```

### 2.2. Độ nhạy với Phép xoay Không gian (Rotational Sensitivity)

Cây quyết định phân tách không gian bằng các đường ranh giới trực giao (Axis-aligned splits) song song với các trục tọa độ.

* **Vấn đề:** Nếu ranh giới phân tách thực tế giữa 2 lớp là một đường chéo (ví dụ: $x_2 = x_1$), Cây quyết định sẽ phải tạo ra hàng loạt phép cắt dọc/ngang nối tiếp nhau dạng bậc thang (staircase pattern).
* **Hậu quả:** Cây trở nên cực kỳ sâu, phức tạp và rất dễ quá khớp.
* **Giải pháp:** Áp dụng thuật toán xoay không gian như PCA (Principal Component Analysis) trước khi đưa dữ liệu vào Cây.

---

## 3. Minimal Cost-Complexity Pruning (Cắt tỉa Tối ưu Chi phí - Tốc độ)

Thay vì chặn cây phát triển sớm (Pre-pruning có thể làm cây bỏ lỡ các phép tách tốt ở tầng sâu hơn), chiến thuật tối ưu hơn là cho cây phát triển tối đa, sau đó tiến hành tỉa cành (Post-pruning).

Scikit-Learn triển khai thuật toán Minimal Cost-Complexity Pruning điều khiển bởi tham số phạt $\alpha \ge 0$ (`ccp_alpha`).

### 3.1. Công thức Toán học

Hàm chi phí đao tạo của cây $T$ khi có tham số phạt $\alpha$:

$$R_\alpha(T) = R(T) + \alpha \vert{}\tilde{T}\vert{}$$

Trong đó:

* $R(T)$: Tổng độ bất thuần (Impurity) hoặc sai số của tất cả các nút lá trong cây $T$.
* $\vert{}\tilde{T}\vert{}$: Tổng số lượng nút lá (Leaf nodes) của cây $T$.
* $\alpha$ (`ccp_alpha`): Tham số phạt độ phức tạp.

### 3.2. Cơ chế hoạt động của $\alpha$

* Khi $\alpha = 0$: Không có hình phạt, cây $T$ giữ nguyên trạng thái fully-grown (nhiều lá nhất, $R(T)$ nhỏ nhất).
* Khi $\alpha$ tăng dần: Chi phí cho mỗi nút lá tăng lên. Thuật toán sẽ tiến hành cắt bỏ các nhánh con có Mức giảm độ bất thuần trên mỗi lá (Effective Alpha) nhỏ hơn $\alpha$.
* Khi $\alpha \to \infty$: Cây bị tỉa sạch chỉ còn đúng $1$ nút gốc.

$$\alpha_{\text{eff}} = \frac{R(t) - R(T_t)}{\vert{}T_t\vert{} - 1}$$

(Trong đó $t$ là nút nội bộ, $T_t$ là cây con có gốc tại $t$)

---

## 4. Mã Nguồn Đầy Đủ Có Thể Chạy: Đường Dẫn Cắt Tỉa & Tinh Chỉnh (Complete Executable Code: Cost-Complexity Pruning Path & Tuning)

Đoạn mã dưới đây minh họa quy trình trích xuất chuỗi $\alpha$ ứng viên và dùng GridSearchCV để tìm giá trị `ccp_alpha` tối ưu.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Tải dữ liệu Ung thư vú
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Huấn luyện cây phát triển tối đa (Fully-grown tree)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 3. Trích xuất đường dẫn cắt tỉa (Pruning Path)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Loại bỏ giá trị alpha lớn nhất vì nó sẽ tỉa sạch cây thành 1 nút gốc
ccp_alphas_trim = ccp_alphas[:-1]

print(f"Tổng số giá trị alpha ứng viên tìm được: {len(ccp_alphas_trim)}")

# 4. Tìm ccp_alpha tối ưu bằng GridSearchCV & 5-Fold Cross Validation
param_grid = {'ccp_alpha': ccp_alphas_trim}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['ccp_alpha']
print(f"--> Tham số ccp_alpha tối ưu nhất: {best_alpha:.6f}")

# 5. Đánh giá hiệu quả trước và sau khi Pruning
# Cây chưa cắt tỉa
unpruned_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
train_acc_unpruned = accuracy_score(y_train, unpruned_model.predict(X_train))
test_acc_unpruned = accuracy_score(y_test, unpruned_model.predict(X_test))

# Cây đã cắt tỉa (Post-pruned)
pruned_model = grid_search.best_estimator_
train_acc_pruned = accuracy_score(y_train, pruned_model.predict(X_train))
test_acc_pruned = accuracy_score(y_test, pruned_model.predict(X_test))

print("\n=== SO SÁNH HIỆU NĂNG BÀI TOÁN ===")
print(f"Chưa tỉa  | Train Acc: {train_acc_unpruned:.4f} | Test Acc: {test_acc_unpruned:.4f} (Chênh lệch: {train_acc_unpruned - test_acc_unpruned:.4f})")
print(f"Đã tỉa    | Train Acc: {train_acc_pruned:.4f} | Test Acc: {test_acc_pruned:.4f} (Chênh lệch: {train_acc_pruned - test_acc_pruned:.4f})")
print(f"Số nút lá | Chưa tỉa: {unpruned_model.get_n_leaves()} lá -> Đã tỉa: {pruned_model.get_n_leaves()} lá")
```

## 5. Production Readiness Checklist (Danh mục kiểm định Triển khai)

Trước khi đưa mô hình Cây quyết định vào hệ thống Production hoặc báo cáo:

- [ ] **Bỏ qua Chuẩn hóa Thang đo:** Xác nhận không vô tình dùng `StandardScaler` (không làm sai nhưng tốn tài nguyên tính toán và làm mất tính minh bạch của ngưỡng $t$).
- [ ] **Kiểm tra Khoảng Ngoại suy:** Nếu bài toán đòi hỏi dự đoán các giá trị tương lai nằm ngoài khoảng lịch sử (ví dụ: dự đoán doanh thu tăng trưởng theo năm), không sử dụng Decision Tree đơn lẻ.
- [ ] **Kiểm soát Overfitting:** Đã áp dụng ít nhất một cơ chế kiểm soát: Pre-pruning (`max_depth`, `min_samples_leaf`) hoặc Post-pruning (`ccp_alpha`).
- [ ] **Xử lý Mất cân bằng Lớp:** Đã cấu hình `class_weight='balanced'` nếu phân phối nhãn chênh lệch đáng kể (ví dụ $> 70:30$).
- [ ] **Kiểm tra Độ sâu Cây:** Đảm bảo cây phục vụ dự đoán có độ sâu vừa phải để không gây tốn dung lượng RAM và giữ thời gian phản hồi (latency) thấp.
- [ ] **Đánh giá Đa phương diện:** Luôn so sánh chênh lệch giữa Train Score và Test Score. Chênh lệch lý tưởng nên $< 5\%$.
