---
topic: "Logistic Regression"
subtopic: "Classification Metrics, Diagnostics & Evaluation Protocols"
level: "Intermediate"
doc_id: "logreg_04"
sources:
  - "Scikit-Learn User Guide v1.4: Model Evaluation"
---

# Classification Evaluation Metrics & Model Diagnostics

## 1. Confusion Matrix Structure

Ma trận nhầm lẫn (Confusion Matrix) là bảng thống kê số lượng dự đoán đúng và sai của mô hình phân loại nhị phân:

| | Dự đoán Negative ($y=0$) | Dự đoán Positive ($y=1$) |
| :--- | :--- | :--- |
| **Thực tế Negative ($y=0$)** | True Negative (**TN**) | False Positive (**FP**) *(Lỗi Type I)* |
| **Thực tế Positive ($y=1$)** | False Negative (**FN**) *(Lỗi Type II)* | True Positive (**TP**) |

---

## 2. Core Numerical Evaluation Metrics

### 2.1. Accuracy (Độ chính xác toàn cục)

Tỉ lệ các mẫu được dự đoán đúng trên tổng số mẫu:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

*Hạn chế*: Mất tác dụng khi tập dữ liệu bị mất cân bằng nhãn.

### 2.2. Precision (Độ chính xác lớp Positive)

Tỉ lệ các mẫu thực sự là Positive trong số các mẫu được mô hình dự đoán là Positive:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

*Ứng dụng*: Cần ưu tiên cao khi chi phí của False Positive quá đắt (ví dụ: Hệ thống lọc Email Spam, cấp duyệt tín dụng ngân hàng).

### 2.3. Recall / Sensitivity (Độ triệu hồi)

Tỉ lệ các mẫu Positive được mô hình phát hiện thành công trong tổng số các mẫu Positive thực tế:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

*Ứng dụng*: Cần ưu tiên cao khi chi phí của False Negative quá đắt (ví dụ: Chẩn đoán bệnh hiểm nghèo, phát hiện gian lận tài chính).

### 2.4. F1-Score

Trung bình điều hòa (Harmonic Mean) giữa Precision và Recall, giúp đánh giá sự cân bằng giữa hai chỉ số này:

$$
F_1 =
2 \cdot
\frac{\text{Precision} \cdot \text{Recall}}
{\text{Precision} + \text{Recall}}
=
\frac{2 \cdot \text{TP}}
{2 \cdot \text{TP} + \text{FP} + \text{FN}}
$$

---

## 3. Diagnostic Curves & Analysis

### 3.1. ROC Curve & ROC-AUC

* **ROC Curve (Receiver Operating Characteristic)**: Đồ thị biểu diễn mối tương quan giữa **True Positive Rate (Recall)** trên trục tung và **False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$)** trên trục hoành khi thay đổi ngưỡng quyết định từ $1.0$ về $0.0$.

* **ROC-AUC Score**: Diện tích nằm dưới đường cong ROC ($0.5 \le \text{AUC} \le 1.0$).

  * $\text{AUC} = 0.5$: Mô hình dự đoán ngẫu nhiên (Random Guessing).
  * $\text{AUC} = 1.0$: Mô hình phân tách hai lớp hoàn hảo.

### 3.2. Precision-Recall (PR) Curve

Khi tập dữ liệu bị mất cân bằng nhãn nghiêm trọng (lớp Positive chiếm tỉ lệ rất nhỏ), đường cong PR trực quan hơn ROC Curve vì FPR trong ROC Curve bị pha loãng bởi số lượng True Negative quá lớn.

---

## 4. Standard Evaluation Routine (Code Python)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score
)

def evaluate_logistic_regression_model(model, X_test, y_test):
    """
    Hàm xuất báo cáo đánh giá toàn diện cho Logistic Regression
    """
    # 1. Dự đoán nhãn và xác suất
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. In Classification Report
    print("=================== BÁO CÁO ĐÁNH GIÁ ===================")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("========================================================")

    # 3. Vẽ ma trận nhầm lẫn và đường cong chẩn đoán
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title("Ma trận nhầm lẫn (Confusion Matrix)")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("Đường cong ROC")

    plt.tight_layout()
    plt.show()
```

---

## 5. Official References

Scikit-Learn Metrics Reference:

https://scikit-learn.org/stable/modules/model_evaluation.html
