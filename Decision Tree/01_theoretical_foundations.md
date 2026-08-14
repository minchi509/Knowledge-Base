---
topic: "Decision Tree"
subtopic: "Theoretical Foundations & Mathematical Formulation"
level: "Intermediate"
doc_id: "dt_01"
source_url: "https://scikit-learn.org/stable/modules/tree.html"
key_concepts:
  - "CART Algorithm"
  - "Gini Impurity"
  - "Entropy"
  - "Information Gain"
  - "Orthogonal Decision Boundaries"
---

# Theoretical Foundations of Decision Trees

## 1. Core Concept & The CART Algorithm
Decision Tree (Cây quyết định) là một mô hình học máy phi tham số (non-parametric) mô phỏng quá trình ra quyết định của con người. Khác với Logistic Regression tìm kiếm một siêu phẳng (hyperplane) chia cắt toàn bộ không gian, Decision Tree phân chia không gian dữ liệu bằng các đường ranh giới trực giao (orthogonal decision boundaries) song song với các trục đặc trưng.

Scikit-Learn sử dụng phiên bản tối ưu của thuật toán **CART (Classification and Regression Trees)**. Thuật toán này xây dựng cây nhị phân: tại mỗi nút nội bộ (internal node), dữ liệu luôn được chia thành đúng hai nhánh (Trái và Phải) dựa trên một câu hỏi (ví dụ: $x_i \le t$).

## 2. Node Splitting Mathematics (Tiêu chí tách nút)
Để quyết định phân tách tại đặc trưng $j$ với ngưỡng $t$, thuật toán CART tính toán hàm mất mát (Impurity) của nút cha và các nút con. Mục tiêu là cực tiểu hóa độ mất mát này.

### 2.1. Gini Impurity (Chỉ số Gini)

Được sử dụng làm mặc định vì tốc độ tính toán nhanh (không chứa hàm logarit). Gini đo lường xác suất một mẫu ngẫu nhiên bị phân loại sai nếu nó được gán nhãn ngẫu nhiên theo phân phối nhãn tại nút đó.

Tại nút $m$ có $K$ lớp, tỉ lệ mẫu thuộc lớp $k$ là $p_{mk}$.

$$H_{\text{Gini}}(m) = \sum_{k=1}^{K} p_{mk} (1 - p_{mk}) = 1 - \sum_{k=1}^{K} p_{mk}^2$$

* Tính chất: Gini bằng $0$ khi nút hoàn toàn thuần nhất (chỉ chứa 1 lớp). Đạt cực đại ở $1 - (1/K)$ khi các lớp phân bố đều nhau.

#### Cài đặt hàm tính Gini Impurity bằng NumPy thủ công (From Scratch)

```python
import numpy as np

def calculate_gini_scratch(y):
    """
    Tính Gini Impurity cho mảng nhãn y
    """
    if len(y) == 0:
        return 0.0
    
    # Tính tỷ lệ xuất hiện p_k của từng lớp
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    # Công thức: Gini = 1 - sum(p_k^2)
    gini = 1.0 - np.sum(probabilities ** 2)
    return gini
```

#### Ví dụ minh họa: Mảng nhãn gồm 4 mẫu lớp 0 và 1 mẫu lớp 1

```python
y_sample = np.array([0, 0, 0, 0, 1])

print(
    f"Gini Impurity thủ công: "
    f"{calculate_gini_scratch(y_sample):.4f}"
)

# Output:
# Gini Impurity thủ công: 0.3200
```

### 2.2. Entropy (Độ hỗn loạn thông tin)

Lấy cảm hứng từ Lý thuyết Thông tin của Shannon, Entropy đo lường mức độ bất định (uncertainty) tại một nút:

$$H_{\text{Entropy}}(m) = - \sum_{k=1}^{K} p_{mk} \log_2(p_{mk})$$

*(Quy ước: $0 \log_2(0) = 0$)*
### 2.2. Entropy (Độ hỗn loạn thông tin)
Lấy cảm hứng từ Lý thuyết Thông tin của Shannon, Entropy đo lường mức độ bất định (uncertainty) tại một nút:
$$H_{\text{Entropy}}(m) = - \sum_{k=1}^{K} p_{mk} \log_2(p_{mk})$$
*(Quy ước: $0 \log_2(0) = 0$)*

## 3. Information Gain & Cost Function
Khi tách một nút cha $m$ thành nút trái $m_L$ và nút phải $m_R$ (với số lượng mẫu tương ứng là $N_m, N_L, N_R$), thuật toán tìm cặp $(j, t)$ để tối thiểu hóa hàm chi phí CART:
$$J(j, t) = \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R)$$
Quá trình này tương đương với việc Tối đa hóa **Information Gain (IG)**:
$$IG = H(m) - \left( \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R) \right)$$
CART áp dụng chiến lược Tham lam (Greedy Strategy): Nó chỉ tìm điểm cắt tốt nhất tại bước hiện tại mà không quan tâm liệu điểm cắt này có dẫn đến cây tối ưu toàn cục ở các bước sau hay không.

## 4. Key Assumptions & Properties
1. **No Scale Assumption**: Thuật toán hoàn toàn vô cảm với thang đo của dữ liệu.
2. **Local Optimization**: Là thuật toán Heuristic, không đảm bảo tìm được cây tối ưu tuyệt đối toàn cục.
3. **Non-linear Relationships**: Tự động nắm bắt các mối quan hệ phi tuyến tính phức tạp giữa đặc trưng và nhãn.
