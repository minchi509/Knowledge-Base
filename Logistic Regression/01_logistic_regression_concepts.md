---
topic: "Logistic Regression"
subtopic: "Theoretical Foundations, Mathematics & Practical Application"
level: "Beginner to Intermediate"
doc_id: "logreg_01"
source_url: "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"
scikit_learn_version: "1.4+"
key_concepts:
  - "phân loại vs hồi quy"
  - "hàm sigmoid"
  - "tỉ số odds và odds ratio"
  - "phép biến đổi logit"
  - "log loss / entropy chéo nhị phân"
  - "tính không lồi của MSE"
  - "điều hòa L1 / L2 / ElasticNet"
  - "tham số C trong scikit-learn"
  - "diễn giải ý nghĩa trọng số"
---

# Logistic Regression

---

## 1. Đặt Vấn Đề: Tại Sao Cần Logistic Regression?

### 1.1. Bài toán Phân loại (Classification)
Trong thực tế, nhiều bài toán không yêu cầu dự đoán một con số liên tục (như giá nhà $500.000$) mà yêu cầu đưa ra quyết định Nhị phân (Binary):
* **Y học:** Bệnh nhân có bị tiểu đường hay không? ($1$: Có, $0$: Không)
* **Tài chính:** Giao dịch này có phải lừa đảo? ($1$: Lừa đảo, $0$: Hợp lệ)
* **Kỹ thuật:** Email này có phải Spam? ($1$: Spam, $0$: Thư thường)

### 1.2. Tại sao Hồi quy Tuyến tính (Linear Regression) thất bại?
Giả sử ta muốn dự đoán khả năng đậu kỳ thi dựa trên số giờ học $x$. Mô hình hồi quy tuyến tính có dạng:
$$\hat{y} = w_1 x + b$$

Cách tiếp cận này gặp **3 sự cố nghiêm trọng**:
1. **Dự đoán nằm ngoài khoảng $[0, 1]$:** Nếu học sinh học $20$ giờ, phương trình có thể cho ra $\hat{y} = 1.5$. Một xác suất bằng $150\%$ là hoàn toàn vô nghĩa.
2. **Nhạy cảm với Điểm ngoại lệ (Outliers):** Thêm một học sinh học $100$ giờ sẽ kéo đường thẳng hồi quy lệch đi hoàn toàn, làm thay đổi ngưỡng phân loại của các dữ liệu còn lại.
3. **Giả định sai về sai số:** Hồi quy tuyến tính giả định sai số có phân phối chuẩn với biến thiên không đổi, điều này không đúng với dữ liệu đầu ra chỉ gồm $\{0, 1\}$.

---

## 2. Hàm Sigmoid: Cầu Nối Biến Đổi Xác Suất

Để giải quyết vấn đề đầu ra văng ra ngoài khoảng $[0, 1]$, ta truyền đầu ra của hàm tuyến tính $z = \mathbf{w}^T \mathbf{x} + b$ qua một hàm phi tuyến có tên là **Hàm Sigmoid** (hay Logistic Function).

### 2.1. Công thức Toán học
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó:
* $z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} + b$ (Đầu ra tuyến tính).
* $e \approx 2.71828$ (Cơ số của logarithm tự nhiên).
* $\sigma(z) \in (0, 1)$ tượng trưng cho xác suất $P(y=1|\mathbf{x})$.

### 2.2. Các Tính chất Hình học quan trọng
* **Khi $z \to +\infty$:** $e^{-z} \to 0 \Rightarrow \sigma(z) \to \frac{1}{1 + 0} = 1$.
* **Khi $z \to -\infty$:** $e^{-z} \to +\infty \Rightarrow \sigma(z) \to \frac{1}{\infty} = 0$.
* **Khi $z = 0$:** $e^0 = 1 \Rightarrow \sigma(0) = \frac{1}{1 + 1} = 0.5$ (**Ngưỡng quyết định - Decision Boundary**).

### 2.3. Ví dụ Tính toán Thủ công từng bước (Dùng cho Bài tập Tính tay)
**Đề bài:** Giả sử mô hình có $w_1 = 0.8$, $b = -2.0$. Xét học sinh học $x = 3$ giờ. Hãy tính xác suất đậu.

* **Bước 1: Tính giá trị tuyến tính $z$**
  $$z = (0.8 \times 3) + (-2.0) = 2.4 - 2.0 = 0.4$$
* **Bước 2: Thay $z$ vào hàm Sigmoid**
  $$\sigma(0.4) = \frac{1}{1 + e^{-0.4}} = \frac{1}{1 + 0.6703} = \frac{1}{1.6703} \approx 0.5986$$
* **Kết luận:** Xác suất học sinh này đậu kỳ thi là **$59.86\%$**. Do $0.5986 \ge 0.5$, mô hình dự đoán nhãn là **$1$ (Đậu)**.

---

## 3. Tỉ Số Odds & Hàm Logit: Bản Chất Của Mô Hình

### 3.1. Tỉ số Odds (Odds Ratio)
Tỉ số Odds là tỉ lệ giữa xác suất sự kiện xảy ra ($p$) và xác suất sự kiện KHÔNG xảy ra ($1 - p$):
$$\text{Odds} = \frac{p}{1 - p}$$

*Ví dụ:* Nếu xác suất đội bóng thắng là $p = 0.8$, thì xác suất thua là $1 - p = 0.2$. Tỉ số $\text{Odds} = \frac{0.8}{0.2} = 4$ (Cơ hội thắng cao gấp 4 lần cơ hội thua).

### 3.2. Hàm Logit (Biến đổi Log-Odds)
Lấy logarithm tự nhiên ($\ln$) của Odds, ta thu được hàm **Logit**:
$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$

Nếu thay $p = \sigma(z) = \frac{1}{1 + e^{-z}}$ vào biến đổi trên, ta có kết quả bất ngờ:
$$\text{logit}(p) = \ln\left( \frac{\frac{1}{1 + e^{-z}}}{1 - \frac{1}{1 + e^{-z}}} \right) = \ln(e^z) = z = \mathbf{w}^T \mathbf{x} + b$$

> **Ý nghĩa Triết học của Logistic Regression:** Logistic Regression thực chất là một mô hình **Hồi quy Tuyến tính dự đoán Log-Odds** của biến mục tiêu.

### 3.3. Giải thích Ý nghĩa Trọng số $w_i$ (Weight Interpretation)
Khi đặc trưng $x_i$ tăng thêm $1$ đơn vị (các đặc trưng khác giữ nguyên):
* Log-Odds tăng thêm đúng $w_i$ đơn vị.
* Tỉ số Odds nhân thêm một hệ số bằng $e^{w_i}$.

*Ví dụ:* Nếu $w_{\text{thu nhập}} = 0.5$, khi thu nhập tăng $1$ đơn vị, cơ hội được duyệt vay tiền tăng lên $e^{0.5} \approx 1.65$ lần ($65\%$).

---

## 4. Hàm Mất Mát Log Loss (Binary Cross-Entropy)

### 4.1. Tại sao KHÔNG thể dùng MSE (Mean Squared Error)?
Nếu dùng MSE cho Logistic Regression: $J(\mathbf{w}) = \frac{1}{N} \sum (\sigma(\mathbf{w}^T \mathbf{x} + b) - y)^2$.
Do chứa hàm phi tuyến Sigmoid bên trong, đồ thị sai số sẽ bị biến dạng tạo thành bề mặt **Không lồi (Non-convex)** với rất nhiều điểm Cực trị Địa phương (Local Minima). Thuật toán Gradient Descent sẽ bị mắc kẹt và không bao giờ tìm được nghiệm tối ưu toàn cục.

### 4.2. Công thức Log Loss
Để tạo bề mặt Lồi (Convex), ta dùng hàm **Log Loss** (dựa trên nguyên lý Maximum Likelihood Estimation - MLE):

$$\text{Loss}(y, \hat{y}) = \begin{cases} 
-\ln(\hat{y}) & \text{nếu } y = 1 \\
-\ln(1 - \hat{y}) & \text{nếu } y = 0 
\end{cases}$$

Viết gộp lại cho $1$ mẫu dữ liệu:
$$\ell(y, \hat{y}) = - \left[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \right]$$

Hàm Mất Mát trung bình trên toàn bộ tập dữ liệu gồm $N$ mẫu:
$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

### 4.3. Cơ chế Trừng phạt của Log Loss
* **Khi $y = 1$:** 
  * Nếu $\hat{y} \to 1$: $-\ln(1) = 0 \Rightarrow$ Phạt bằng $0$.
  * Nếu $\hat{y} \to 0$: $-\ln(\text{số rất nhỏ}) \to +\infty \Rightarrow$ Phạt **cực nặng** (vô cùng).
* **Bài học rút ra:** Log Loss trừng phạt rất đau những dự đoán "vừa sai vừa tự tin".

---
## 5. Kỹ Thuật Điều Hòa (Regularization): Chống Quá Khớp

Khi dữ liệu phân tách hoàn hảo (Linearly Separable), trọng số $\mathbf{w}$ có xu hướng tiến ra vô cùng ($\pm \infty$) khiến mô hình bị Quá khớp (Overfitting). Ta dùng Kỹ thuật Điều hòa để kìm hãm độ lớn của $\mathbf{w}$.

### 5.1. Các loại Regularization

1. **L2 Regularization (Ridge):**

$$J_{\text{L2}}(\mathbf{w}) = J(\mathbf{w}) + \frac{1}{2C} \|\mathbf{w}\|_2^2 = J(\mathbf{w}) + \frac{1}{2C} \sum_{j=1}^{n} w_j^2$$

* **Tác dụng:** Thu nhỏ các trọng số về sát $0$, làm mịn mô hình.

2. **L1 Regularization (Lasso):**

$$J_{\text{L1}}(\mathbf{w}) = J(\mathbf{w}) + \frac{1}{C} \|\mathbf{w}\|_1 = J(\mathbf{w}) + \frac{1}{C} \sum_{j=1}^{n} |w_j|$$

* **Tác dụng:** Có khả năng triệt tiêu trọng số về đúng bằng $0$. Tự động lựa chọn đặc trưng (Feature Selection).

3. **ElasticNet:** 
* Kết hợp cả L1 và L2 theo tỉ lệ `l1_ratio` ($\rho$).

### 5.2. Tham số $C$ trong Scikit-Learn

Trong thư viện `scikit-learn`, $C$ là **hằng số nghịch đảo** của hệ số phạt $\lambda$ ($C = \frac{1}{\lambda}$):
* **$C$ RẤT LỚN (ví dụ $C = 1000$):** Phạt nhẹ $\rightarrow$ Mô hình phức tạp $\rightarrow$ Dễ bị **Overfitting**.
* **$C$ RẤT NHỎ (ví dụ $C = 0.01$):** Phạt mạnh $\rightarrow$ Ép trọng số nhỏ lại $\rightarrow$ Dễ bị **Underfitting**.

---

## 6. Các Giả Định Bắt Buộc (Key Assumptions)

Để áp dụng Logistic Regression đạt hiệu quả cao, tập dữ liệu cần thỏa mãn 4 điều kiện:
1. **Đầu ra mang bản chất Phân loại:** Biến mục tiêu phải thuộc dạng Binary ($0/1$) hoặc Categorical.
2. **Độc lập giữa các quan sát:** Các mẫu dữ liệu không được phụ thuộc lẫn nhau (không dùng cho chuỗi thời gian chưa qua xử lý).
3. **Không có Đa cộng tuyến nghiêm trọng (No Severe Multicollinearity):** Các biến độc lập $x_i, x_j$ không được có tương quan quá mạnh với nhau ($r > 0.9$). Kiểm tra bằng hệ số VIF ($VIF < 5$).
4. **Tuyến tính với Log-Odds:** Các đặc trưng liên tục $x_i$ phải có mối quan hệ tuyến tính với $\ln(\text{Odds})$.

---

## 7. Khung Dẫn Dắt Sinh Bài Tập (Exercise Framework)

Hệ thống có thể tạo ra 3 dạng bài tập dựa trên nội dung tài liệu này:

### Dạng 1: Bài tập Tính toán Thủ công (Hand Calculation)
* **Yêu cầu:** Cho bộ trọng số $\mathbf{w} = [0.5, -1.2]$, $b = 0.3$ và dữ liệu $x = [2.0, 1.5]$.
* **Nhiệm vụ:**
  1. Tính giá trị $z$.
  2. Tính xác suất $\hat{y} = \sigma(z)$.
  3. Tính Log Loss nếu nhãn thực tế $y = 1$.

### Dạng 2: Bài tập Giải thích Mô hình (Interpretation Task)
* **Yêu cầu:** Mô hình dự đoán khả năng mua hàng có $w_{\text{tuổi}} = 0.04$.
* **Nhiệm vụ:** Hãy giải thích bằng lời: Khi khách hàng tăng thêm $10$ tuổi, Tỉ số Odds mua hàng thay đổi bao nhiêu phần trăm? ($e^{0.04 \times 10} = e^{0.4} \approx 1.4918 \Rightarrow$ Tăng $49.18\%$).

### Dạng 3: Bài tập Thực hành Python Pipeline (Coding Project)
* **Yêu cầu:** Lập trình pipeline hoàn chỉnh trên tập dữ liệu thực tế (`breast_cancer` hoặc `titanic`).
* **Các bước phải triển khai:**
  1. Chuẩn hóa dữ liệu bằng `StandardScaler` (Rất quan trọng vì Logistic Regression nhạy cảm với thang đo).
  2. Khởi tạo `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')`.
  3. Đánh giá mô hình bằng các độ đo: Precision, Recall, F1-Score, ROC-AUC.
  4. Trích xuất thuộc tính `.coef_` và vẽ biểu đồ thanh (Bar chart) thể hiện mức độ quan trọng của các đặc trưng.
