---
topic: "Decision Tree"
subtopic: "Common Pitfalls, Overfitting & Cost-Complexity Pruning"
level: "Intermediate"
doc_id: "dt_05"
sources:
  - "Scikit-Learn Minimal Cost-Complexity Pruning"
key_concepts:
  - "Overfitting"
  - "Extrapolation Limitations"
  - "Cost-Complexity Pruning (ccp_alpha)"
  - "Production Checklist"
---

# Common Pitfalls & Pruning Practices

## 1. High Variance & Overfitting Syndrome
**Vấn đề**: Cây quyết định cực kỳ nhạy cảm với tập huấn luyện. Sự thay đổi nhỏ bé của dữ liệu (thêm/bớt 1% số lượng dòng) có thể làm thay đổi hoàn toàn gốc rễ của cây, dẫn đến cấu trúc cây biến đổi toàn diện.
**Triệu chứng điển hình**: Mô hình đạt Accuracy 99% - 100% trên tập Train, nhưng rớt thảm hại xuống 60% trên tập Test.
**Khắc phục**: 
- Sử dụng cơ chế Pre-pruning: Chủ động chặn đứng quá trình mọc rễ bằng `max_depth`, `min_samples_split`.
- Nâng cấp lên thuật toán Ensemble (Random Forest, XGBoost).

## 2. Extrapolation Blindspot (Điểm mù ngoại suy)
**Vấn đề cốt tử**: Decision Tree tuyệt đối **không có khả năng ngoại suy (Extrapolate)**.
Khác với Linear/Logistic Regression, khi bạn nhập vào một mức `Thu_nhap` lớn gấp đôi giá trị lớn nhất từng xuất hiện trong tập Train, mô hình hồi quy tuyến tính sẽ phóng dốc lên cao. Ngược lại, Cây quyết định chỉ đơn giản đưa mẫu đó vào nút lá kề cạnh nhất và trả ra một hằng số dự đoán y hệt như những mẫu từng nằm trong lá đó. Cây không hiểu được "xu hướng tăng tiến".

## 3. Minimal Cost-Complexity Pruning (Cắt tỉa tối ưu)
Thay vì chặn trước (Pre-pruning) mang tính phỏng đoán, ta để cây mọc tối đa, sau đó áp dụng toán học để tỉa cành (Post-pruning) bằng tham số `ccp_alpha`.
Hàm chi phí của cây $T$ được định nghĩa lại:
$$R_\alpha(T) = R(T) + \alpha \vert{}\tilde{T}\vert{}$$
Trong đó:
- $R(T)$: Tổng sai số phân loại của cây.
- $\vert{}\tilde{T}\vert{}$: Số lượng nút lá.
- $\alpha$: Trọng số phạt (Cost-complexity parameter). Càng tăng $\alpha$, các nhánh con mang lại ít Information Gain hơn so với sự phức tạp của chúng sẽ bị cắt bỏ.

```python
# 1. Trích xuất chuỗi các giá trị alpha tiềm năng
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1] # Bỏ giá trị alpha lớn nhất (làm cây chỉ còn 1 nút gốc)

# 2. Bạn có thể dùng GridSearchCV để tìm ccp_alpha tốt nhất
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={'ccp_alpha': ccp_alphas},
    cv=5
)
grid.fit(X_train, y_train)
print(f"Alpha tốt nhất để tỉa cây: {grid.best_params_['ccp_alpha']}")
```
## 4. Production Readiness Checklist

Trước khi đưa mô hình Cây Quyết định vào Notebook bài tập/thực tế:

[ ] Xác nhận không vô tình áp dụng StandardScaler (tuy không làm sai kết quả nhưng làm chậm pipeline và mất tính tường minh).

[ ] Chắc chắn đã kiểm tra hiện tượng Overfitting bằng cách so sánh (Train Score vs Test Score).

[ ] Xác nhận max_depth $\le 10$ hoặc đã áp dụng ccp_alpha để cây không quá tải RAM.

[ ] Đã sử dụng class_weight='balanced' nếu phân phối nhãn bị chênh lệch lớn hơn tỷ lệ 70-30.
