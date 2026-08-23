---
topic: "Logistic Regression"
subtopic: "Classification Metrics, Diagnostics & Evaluation Protocols"
level: "Intermediate"
doc_id: "logreg_04"
source_url: "https://scikit-learn.org/stable/modules/model_evaluation.html"
scikit_learn_version: "1.4"
key_concepts:
  - "Ma trận nhầm lẫn (TP, TN, FP, FN)"
  - "Accuracy, Precision, Recall, F1-Score"
  - "Đường cong ROC & Chỉ số ROC-AUC"
  - "Đường cong Precision-Recall (PR)"
  - "Tự tính toán chỉ số từ đầu bằng NumPy"
  - "Quy trình đánh giá chuẩn với Scikit-Learn & Matplotlib"
---

# Đánh Giá Mô Hình & Chẩn Đoán Phân Loại Cho Logistic Regression

---

## 1. Ma Trận Nhầm Lẫn (Confusion Matrix)

Ma trận nhầm lẫn (Confusion Matrix) là bảng thống kê số lượng kết quả dự đoán đúng và sai của mô hình phân loại nhị phân:

| | Dự đoán Negative ($y=0$) | Dự đoán Positive ($y=1$) |
| --- | --- | --- |
| **Thực tế Negative ($y=0$)** | True Negative (TN) | False Positive (FP) — *Lỗi Type I* |
| **Thực tế Positive ($y=1$)** | False Negative (FN) — *Lỗi Type II* | True Positive (TP) |

- **True Positive (TP):** Mẫu thực tế thuộc lớp $1$ và mô hình dự đoán đúng là $1$.
- **True Negative (TN):** Mẫu thực tế thuộc lớp $0$ và mô hình dự đoán đúng là $0$.
- **False Positive (FP - Lỗi Type I):** Mẫu thực tế thuộc lớp $0$ nhưng mô hình dự đoán sai thành $1$.
- **False Negative (FN - Lỗi Type II):** Mẫu thực tế thuộc lớp $1$ nhưng mô hình dự đoán sai thành $0$.

---

## 2. Các Chỉ Số Đánh Giá Định Lượng Cốt Lõi

### 2.1. Accuracy (Độ chính xác toàn cục)

Tỉ lệ các mẫu được dự đoán đúng trên tổng số mẫu trong tập dữ liệu:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**Hạn chế:** Chỉ số này hoàn toàn bị mất tác dụng khi tập dữ liệu bị mất cân bằng nhãn nặng (ví dụ: $99\%$ mẫu là $0$ và $1\%$ mẫu là $1$, mô hình chỉ cần dự đoán toàn $0$ cũng đạt Accuracy $99\%$).

### 2.2. Precision (Độ chính xác lớp Positive)

Tỉ lệ các mẫu thực sự là Positive trong số những mẫu mà mô hình đã dự đoán là Positive:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**Ứng dụng thực tế:** Cần ưu tiên khi chi phí xảy ra lỗi False Positive quá đắt.

*Ví dụ:* Hệ thống lọc Email Spam (bỏ nhầm email công việc quan trọng vào hòm thư Rác) hoặc Cấp duyệt tín dụng tự động.

### 2.3. Recall / Sensitivity (Độ triệu hồi)

Tỉ lệ các mẫu Positive mà mô hình phát hiện thành công trên tổng số mẫu Positive thực tế:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**Ứng dụng thực tế:** Cần ưu tiên khi chi phí xảy ra lỗi False Negative quá đắt.

*Ví dụ:* Chẩn đoán bệnh hiểm nghèo (bỏ sót bệnh nhân ung thư) hoặc Phát hiện giao dịch gian lận tài chính.

### 2.4. F1-Score

Trung bình điều hòa (Harmonic Mean) giữa Precision và Recall, giúp đo lường sự cân bằng giữa hai chỉ số này:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}
$$

---

## 3. Các Đường Cong Chẩn Đoán (Diagnostic Curves)

### 3.1. Đường Cong ROC & Chỉ Số ROC-AUC

**ROC Curve (Receiver Operating Characteristic):** Đồ thị biểu diễn mối tương quan giữa True Positive Rate (Recall) trên trục tung và False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$) trên trục hoành khi thay đổi ngưỡng quyết định (Decision Threshold) từ $1.0$ về $0.0$.

**ROC-AUC Score:** Diện tích nằm dưới đường cong ROC ($0.5 \le \text{AUC} \le 1.0$):

- $\text{AUC} = 0.5$: Mô hình dự đoán ngẫu nhiên (Random Guessing).
- $\text{AUC} = 1.0$: Mô hình phân tách hai lớp hoàn hảo.

### 3.2. Đường Cong Precision-Recall (PR Curve)

Khi tập dữ liệu bị mất cân bằng nhãn nghiêm trọng (lớp Positive chiếm tỉ lệ cực nhỏ, ví dụ $0.1\%$), đường cong PR phản ánh hiệu năng thực tế chính xác hơn ROC Curve. Lý do là vì chỉ số FPR trong ROC Curve bị pha loãng bởi lượng lớn mẫu True Negative.

---

## 4. Cài Đặt Thủ Công Bằng NumPy (Scratch Implementation)

Tự cài đặt các chỉ số bằng NumPy giúp nắm vững thuật toán mà không cần phụ thuộc vào `sklearn.metrics`.

### 4.1. Tự tính Confusion Matrix và các chỉ số cơ bản

```python
import numpy as np

def confusion_matrix_scratch(y_true, y_pred):
    """
    Tự đếm TP, TN, FP, FN từ 2 mảng nhãn 0/1 bằng NumPy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN


def classification_metrics_scratch(y_true, y_pred):
    """
    Tự tính Accuracy, Precision, Recall, F1 theo công thức toán
    """
    TP, TN, FP, FN = confusion_matrix_scratch(y_true, y_pred)
    epsilon = 1e-15  # Tránh lỗi chia cho 0

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
```

### 4.2. Tự tính ROC Curve và ROC-AUC (Thuật toán quét ngưỡng)

Vì ROC-AUC không có công thức đóng (Closed-form expression), ta thực hiện quét qua nhiều ngưỡng quyết định từ $1.0$ về $0.0$, tính cặp điểm (FPR, TPR) tại mỗi ngưỡng và tính diện tích hình thang (Trapezoidal Rule):

```python
def roc_curve_scratch(y_true, y_proba, n_thresholds=100):
    """
    Tự tính các điểm (FPR, TPR) tương ứng với n_thresholds ngưỡng quyết định
    """
    y_true = np.asarray(y_true)
    thresholds = np.linspace(1, 0, n_thresholds)

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        TP, TN, FP, FN = confusion_matrix_scratch(y_true, y_pred_t)

        tpr = TP / (TP + FN + 1e-15)  # TPR = Recall
        fpr = FP / (FP + TN + 1e-15)  # FPR

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_scratch(y_true, y_proba, n_thresholds=100):
    """
    Tính diện tích dưới đường cong ROC bằng quy tắc tích phân hình thang
    """
    fpr, tpr, _ = roc_curve_scratch(y_true, y_proba, n_thresholds)

    # Sắp xếp FPR tăng dần trước khi tính diện tích tích phân
    order = np.argsort(fpr)
    auc = np.trapz(tpr[order], fpr[order])
    return auc
```

### 4.3. Bảng Đối Chiếu So Sánh Scratch vs Scikit-Learn

| Chỉ số | Hàm tự cài đặt (Scratch) | Hàm Scikit-Learn | Khác biệt chính |
| --- | --- | --- | --- |
| Metrics | `classification_metrics_scratch()` | `accuracy_score()`, `precision_score()`, ... | Bản Scratch tính toán trực tiếp trên mảng NumPy 1D. |
| ROC Curve | `roc_curve_scratch()` | `sklearn.metrics.roc_curve` | Sklearn tự chọn các ngưỡng điểm uốn thay vì rải đều `n_thresholds`. |
| ROC-AUC | `roc_auc_scratch()` | `sklearn.metrics.roc_auc_score` | Độ chính xác bản Scratch phụ thuộc vào số lượng ngưỡng `n_thresholds`. |

---

## 5. Quy Trình Đánh Giá Thực Chiến (Standard Evaluation Routine)

Trong công việc thực tế, ta viết hàm đánh giá tự động kết hợp trực quan hóa bằng `matplotlib` và `sklearn.metrics`:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score
)

def evaluate_logistic_regression_model(model, X_test, y_test):
    """
    Hàm xuất báo cáo đánh giá toàn diện cho mô hình Logistic Regression
    """
    # 1. Dự đoán nhãn và xác suất
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. In báo cáo văn bản
    print("=================== BÁO CÁO ĐÁNH GIÁ ===================")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("========================================================")

    # 3. Vẽ biểu đồ Ma trận nhầm lẫn và Đường cong ROC
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title("Ma trận nhầm lẫn (Confusion Matrix)")

    # Đường cong ROC
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("Đường cong ROC")

    plt.tight_layout()
    plt.show()
```

---

## 6. Khung Dẫn Dắt Bài Tập & Thực Hành (Exercise Framework)

### Dạng 1: Tự viết hàm PR Curve (Scratch)

- **Yêu cầu:** Dựa trên hàm `roc_curve_scratch`, hãy tự viết hàm `precision_recall_curve_scratch(y_true, y_proba)` để trả về 2 mảng Precision và Recall tương ứng với các ngưỡng quyết định.

### Dạng 2: Phân tích Bài toán Thực tế (Business Metric Selection)

- **Yêu cầu:** Cho 2 bài toán phân loại:
  1. Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
  2. Phân loại bình luận tiêu cực trên mạng xã hội (Toxic Comment Detection).
- **Câu hỏi:** Bài toán nào nên tối ưu Recall, bài toán nào nên tối ưu Precision? Giải thích lý do dựa trên chi phí của lỗi FP và FN.

---

## 7. Tài Liệu Tham Khảo (Official References)

- Scikit-Learn Model Evaluation Guide: [Model Evaluation Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
