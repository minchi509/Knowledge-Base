---
topic: "Decision Tree"
subtopic: "Theoretical Foundations & Mathematical Formulation"
level: "Intermediate"
doc_id: "dt_01"
source_url: "https://scikit-learn.org/stable/modules/tree.html"
key_concepts:
  - "Thuật toán CART"
  - "Chỉ số Gini (Gini Impurity)"
  - "Entropy (Độ hỗn loạn thông tin)"
  - "Mức tăng thông tin (Information Gain)"
  - "Đường ranh giới quyết định trực giao"
---

# Nền Tảng Lý Thuyết Của Cây Quyết Định (Decision Trees)

## 1. Khái Niệm Cốt Lõi & Thuật Toán CART

### 1.1. Ý Tưởng Cốt Lõi

Decision Tree (Cây quyết định) là một mô hình học máy phi tham số (non-parametric) mô phỏng quá trình tư duy ra quyết định dạng sơ đồ cây (Flowchart) của con người.

- **Ví dụ thực tế:** Khi quyết định "Có nên đi chơi bóng đá hay không?", bạn sẽ tự đặt chuỗi câu hỏi: *Thời tiết thế nào?* $\rightarrow$ *Trời mưa* $\rightarrow$ *Có sân trong nhà không?* $\rightarrow$ *Có* $\rightarrow$ *Đi đá bóng*.
- **So sánh với Logistic Regression:** Trong khi Logistic Regression cố gắng vẽ một đường thẳng/siêu phẳng (hyperplane) mềm mại để phân tách dữ liệu, Decision Tree chia nhỏ không gian bằng các **đường ranh giới trực giao (orthogonal decision boundaries)** — tức các đường cắt vuông góc và song song tuyệt đối với các trục tọa độ.

### 1.2. Thuật toán CART

Scikit-Learn sử dụng phiên bản tối ưu của thuật toán **CART (Classification and Regression Trees)**.

- **Cấu trúc:** CART luôn xây dựng **cây nhị phân (Binary Tree)**. Tại mỗi nút nội bộ (internal node), dữ liệu chỉ được chia thành đúng 2 nhánh (Trái và Phải) dựa trên một câu hỏi điều kiện dạng $x_i \le t$.
- **Cách hoạt động:** Bắt đầu từ nút gốc (Root Node) chứa toàn bộ dữ liệu, thuật toán tìm câu hỏi tốt nhất để chia dữ liệu thành 2 phần "thuần khiết" hơn, sau đó lặp lại đệ quy cho đến khi đạt điều kiện dừng.

---

## 2. Toán Học Chia Nút (Tiêu Chí Chia Nút)

Để chọn ra câu hỏi phân tách tốt nhất tại đặc trưng $j$ với ngưỡng $t$, CART cần đo lường **Độ bất thuần (Impurity)** — tức mức độ xáo trộn/lộn xộn của các nhãn dữ liệu tại một nút.

- **Nút thuần khiết (Pure Node):** Chứa $100\%$ mẫu thuộc về $1$ lớp → Impurity = $0$.
- **Nút không thuần khiết (Impure Node):** Chứa các mẫu phân bổ đều giữa các lớp → Impurity đạt cực đại.

### 2.1. Chỉ Số Gini (Gini Impurity)

Gini Impurity là tiêu chuẩn mặc định trong Scikit-Learn nhờ tốc độ tính toán siêu nhanh (không chứa phép tính logarit phức tạp). Gini đo xác suất một mẫu ngẫu nhiên bị phân loại sai nếu ta gán nhãn ngẫu nhiên theo phân phối nhãn hiện có tại nút.

Tại nút $m$ có $K$ lớp, tỉ lệ mẫu thuộc lớp $k$ là $p_{mk}$:

$$
H_{\text{Gini}}(m) = \sum_{k=1}^{K} p_{mk} (1 - p_{mk}) = 1 - \sum_{k=1}^{K} p_{mk}^2
$$

**Ý nghĩa giá trị:**

- $H_{\text{Gini}} = 0$: Nút hoàn toàn thuần nhất (chỉ chứa 1 lớp nhãn).
- $H_{\text{Gini}} = 1 - \frac{1}{K}$: Nút hỗn loạn nhất (ví dụ với phân loại nhị phân $K=2$, giá trị cực đại là $0.5$ khi tỷ lệ là 50%-50%).

#### Tự Cài Đặt Tính Gini Impurity Bằng NumPy

```python
import numpy as np

def calculate_gini_scratch(y: np.ndarray) -> float:
    """
    Tính Gini Impurity cho mảng nhãn y
    """
    if len(y) == 0:
        return 0.0

    # Tính tỷ lệ xuất hiện p_k của từng lớp nhãn
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    # Công thức: Gini = 1 - sum(p_k^2)
    gini = 1.0 - np.sum(probabilities ** 2)
    return float(gini)

# Minh họa: Mảng gồm 4 mẫu lớp 0 và 1 mẫu lớp 1
y_sample = np.array([0, 0, 0, 0, 1])
print(f"Gini Impurity: {calculate_gini_scratch(y_sample):.4f}")
# Output: Gini Impurity: 0.3200
```

### 2.2. Độ Hỗn Loạn Thông Tin (Entropy)

Lấy cảm hứng từ Lý thuyết Thông tin của Claude Shannon, Entropy đo lường mức độ bất định hoặc lượng thông tin thiếu hụt tại một nút:

$$
H_{\text{Entropy}}(m) = - \sum_{k=1}^{K} p_{mk} \log_2(p_{mk})
$$

(Quy ước toán học: $0 \cdot \log_2(0) = 0$)

**Ý nghĩa giá trị:** Với bài toán phân loại nhị phân ($K=2$), Entropy dao động trong khoảng từ $0$ (thuần khiết tuyệt đối) đến $1.0$ (hỗn loạn tối đa tại tỷ lệ 50%-50%).

#### Tự Cài Đặt Tính Entropy Bằng NumPy

```python
def calculate_entropy_scratch(y: np.ndarray) -> float:
    """
    Tính Entropy cho mảng nhãn y
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    # Bỏ qua các p_k = 0 để tránh lỗi log2(0)
    probabilities = probabilities[probabilities > 0]

    # Công thức: Entropy = -sum(p_k * log2(p_k))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)

# Minh họa: Mảng gồm 4 mẫu lớp 0 và 1 mẫu lớp 1
print(f"Entropy: {calculate_entropy_scratch(y_sample):.4f}")
# Output: Entropy: 0.7219
```

---

## 3. Mức Tăng Thông Tin & Hàm Chi Phí

### 3.1. Mức Tăng Thông Tin (Information Gain)

Khi tách nút cha $m$ thành nút trái $m_L$ và nút phải $m_R$ (với số lượng mẫu tương ứng là $N_m, N_L, N_R$), ta muốn nút con thu được phải ít hỗn loạn hơn nút cha.

Information Gain (IG) đo lường lượng độ bất thuần bị giảm đi sau phép tách:

$$
IG(m, j, t) = H(m) - \left( \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R) \right)
$$

Trong đó $H(m)$ có thể là $H_{\text{Gini}}$ hoặc $H_{\text{Entropy}}$.

### 3.2. Hàm chi phí của CART (Cost Function)

Thuật toán CART duyệt qua tất cả các đặc trưng $j$ và tất cả các ngưỡng cắt $t$ có thể có để **Tối đa hóa Information Gain**, tương đương với việc **Tối thiểu hóa Hàm chi phí** $J(j, t)$:

$$
J(j, t) = \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R)
$$

**Chiến lược Tham lam (Greedy Strategy):** Tại mỗi bước, CART chỉ chọn phép cắt tốt nhất ở thời điểm hiện tại mà không tính đến việc phép cắt đó có mang lại cây tối ưu toàn cục ở các bước sau hay không.

---

## 4. Bảng So Sánh Gini Impurity vs Entropy

| Tiêu chí | Gini Impurity | Entropy |
| --- | --- | --- |
| Công thức | $1 - \sum p_k^2$ | $-\sum p_k \log_2(p_k)$ |
| Tốc độ tính toán | Nhanh hơn (chỉ dùng phép nhân/trừ) | Chậm hơn (phải tính hàm $\log_2$) |
| Khoảng giá trị ($K=2$) | $0.0 \rightarrow 0.5$ | $0.0 \rightarrow 1.0$ |
| Cân bằng nhánh | Có xu hướng tách ra các nhánh chứa lớp phổ biến nhất | Có xu hướng tạo ra các cây cân bằng hơn một chút |
| Ứng dụng thực tế | Mặc định trong sklearn (tốt cho hầu hết trường hợp) | Dùng khi muốn phân tích kỹ lượng thông tin |

---

## 5. Các Giả Định & Tính Chất Cốt Lõi

- **Vô cảm với thang đo (No Scale Assumption):** Decision Tree không phụ thuộc vào độ lớn dữ liệu. Bạn không cần chuẩn hóa dữ liệu (Standardization/Normalization) hay Min-Max Scaling trước khi huấn luyện.
- **Tối ưu cục bộ (Local Optimization):** Do dùng thuật toán tham lam (Greedy), cây có thể bị bẫy ở giải pháp tối ưu cục bộ thay vì cây tối ưu tuyệt đối (NP-complete problem).
- **Mối quan hệ phi tuyến (Non-linear Relationships):** Tự động học các mối quan hệ phức tạp, phi tuyến tính giữa đặc trưng và nhãn mà không cần biến đổi đặc trưng (feature engineering) phức tạp.
- **Dễ bị quá khớp (High Risk of Overfitting):** Nếu không giới hạn độ sâu, cây sẽ phát triển vô tận để nhớ sạch toàn bộ tập train, dẫn đến mô hình hoạt động rất kém trên tập test.
