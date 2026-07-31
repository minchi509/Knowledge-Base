---
topic: "Logistic Regression"
subtopic: "Theoretical Foundations & Mathematical Formulation"
level: "Beginner/Intermediate"
doc_id: "logreg_01"
source_url: "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"
scikit_learn_version: "1.4+"
---

# Theoretical Foundations of Logistic Regression

## 1. Core Concept & Sigmoid Function
Logistic Regression là mô hình học máy có giám sát (Supervised Learning) dùng cho bài toán phân loại (Classification). Khác với Linear Regression dự đoán giá trị liên tục $y \in \mathbb{R}$, Logistic Regression dự đoán xác suất $P(y=1\vert{}\mathbf{x}) \in (0, 1)$ thuộc về lớp dương (positive class).

Để chuyển đổi kết quả của một hàm tuyến tính sang khoảng xác suất từ 0 đến 1, mô hình sử dụng **Sigmoid Function** (hay Logistic Function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó, $z$ là sự kết hợp tuyến tính giữa các đặc trưng (features) và trọng số (weights):

$$z = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n = \mathbf{w}^T \mathbf{x} + b$$

### Properties of Sigmoid Function
* Nếu $z \to +\infty$, $\sigma(z) \to 1$.
* Nếu $z \to -\infty$, $\sigma(z) \to 0$.
* Tại $z = 0$, $\sigma(0) = 0.5$ (ngưỡng quyết định mặc định).

## 2. Odds Ratio & Logit Transformation
Tỉ số Odds (Odds Ratio) phản ánh tỉ lệ giữa xác suất xảy ra sự kiện $p$ và xác suất không xảy ra sự kiện $(1-p)$:

$$\text{Odds} = \frac{p}{1 - p}$$

Lấy logarithm tự nhiên của Odds thu được hàm **Logit**:

$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b$$

Ý nghĩa: Logistic Regression bản chất là một mô hình tuyến tính đối với **Log-Odds** của biến mục tiêu.

## 3. Loss Function (Binary Cross-Entropy / Log Loss)
Mô hình không sử dụng Mean Squared Error (MSE) làm hàm mất mát vì MSE tạo ra một bề mặt tối ưu không lồi (non-convex) có nhiều cực trị địa phương. Thay vào đó, Logistic Regression áp dụng hàm **Log Loss** dựa trên nguyên lý Maximum Likelihood Estimation (MLE):

$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

Trong đó:
* $N$: Số lượng mẫu dữ liệu.
* $y_i \in \{0, 1\}$: Nhãn thực tế của mẫu thứ $i$.
* $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$: Xác suất dự đoán mẫu $i$ thuộc lớp 1.

## 4. Regularization Techniques
Để ngăn chặn tình trạng quá khớp (overfitting), phạt trọng số (penalty) được cộng thêm vào hàm mất mát:

* **L2 Regularization (Ridge)**: Phạt theo bình phương độ lớn trọng số $\frac{1}{2C} \Vert{}\mathbf{w}\Vert{}_2^2$. Giúp thu nhỏ trọng số về gần 0.
* **L1 Regularization (Lasso)**: Phạt theo giá trị tuyệt đối trọng số $\frac{1}{C} \Vert{}\mathbf{w}\Vert{}_1$. Có khả năng triệt tiêu trọng số về đúng 0 (Feature Selection).
* **ElasticNet**: Kết hợp cả L1 và L2 với tỉ lệ điều phối `l1_ratio`.

*Lưu ý: Trong scikit-learn, hằng số $C$ nghịch đảo với độ mạnh điều hòa. $C$ nhỏ tương ứng với điều hòa mạnh.*

## 5. Key Assumptions
1. **Binary or Categorical Target**: Biến phụ thuộc phải mang bản chất phân loại.
2. **Independence of Observations**: Các mẫu dữ liệu độc lập với nhau.
3. **No Severe Multicollinearity**: Các đặc trưng độc lập không có tương quan quá mạnh với nhau.
4. **Linearity in Log-Odds**: Mối quan hệ giữa các đặc trưng liên tục và Log-Odds phải mang tính tuyến tính.

## 6. References
* Scikit-Learn Official User Guide: Linear Models - Logistic Regression
* Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.3.2.

