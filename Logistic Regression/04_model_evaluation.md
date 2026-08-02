---
topic: "Logistic Regression"
subtopic: "Classification Metrics, Diagnostics & Evaluation Protocols (kèm cài đặt thủ công bằng NumPy)"
level: "Intermediate"
doc_id: "logreg_04"
sources:
  - "Tự triển khai các công thức toán ở mục 1, 2, 3 của chính tài liệu này"
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

## 4. Cài Đặt Thủ Công Các Chỉ Số Đánh Giá (Không Dùng `sklearn.metrics`)

### 4.1. Vì sao cần tự tính tay?

Ở mục 5 (Standard Evaluation Routine), ta sẽ gọi thẳng `classification_report`, `confusion_matrix`, `roc_auc_score` từ `sklearn.metrics`. Các hàm này chỉ tốn 1 dòng code nhưng lại **che giấu** cách các chỉ số Accuracy, Precision, Recall, F1, ROC-AUC ở mục 2 và 3 thực sự được tính ra sao từ TP, TN, FP, FN. Phần dưới đây cài đặt lại toàn bộ chỉ bằng NumPy để đối chiếu trực tiếp với công thức toán đã nêu.

### 4.2. Tự tính Confusion Matrix và các chỉ số cơ bản

```python
import numpy as np

def confusion_matrix_scratch(y_true, y_pred):
    """
    Tự đếm TP, TN, FP, FN từ 2 mảng nhãn 0/1 mà không dùng sklearn.
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
    Tự tính Accuracy, Precision, Recall, F1 đúng theo công thức ở mục 2.
    """
    TP, TN, FP, FN = confusion_matrix_scratch(y_true, y_pred)
    epsilon = 1e-15  # tránh chia cho 0

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

| Biến trong code | Công thức tương ứng (mục 1, 2) |
| :--- | :--- |
| `TP, TN, FP, FN` | Bốn ô trong Confusion Matrix ở mục 1 |
| `accuracy` | $\text{Accuracy} = \dfrac{TP+TN}{TP+TN+FP+FN}$ (mục 2.1) |
| `precision` | $\text{Precision} = \dfrac{TP}{TP+FP}$ (mục 2.2) |
| `recall` | $\text{Recall} = \dfrac{TP}{TP+FN}$ (mục 2.3) |
| `f1` | $F_1 = \dfrac{2 \cdot TP}{2 \cdot TP + FP + FN}$ (mục 2.4) |

### 4.3. Tự tính ROC Curve và ROC-AUC (thuật toán quét ngưỡng)

ROC-AUC không có một công thức đóng (closed-form) đơn giản để tính trực tiếp từ dữ liệu — cách nó được tính thực chất là **quét qua nhiều ngưỡng quyết định** (threshold) từ $1.0$ về $0.0$, tại mỗi ngưỡng tính lại TPR và FPR, sau đó lấy diện tích dưới đường cong bằng quy tắc hình thang (Trapezoidal Rule):

```python
def roc_curve_scratch(y_true, y_proba, n_thresholds=100):
    """
    Tự tính các điểm (FPR, TPR) tương ứng với mỗi ngưỡng quyết định,
    thay vì gọi sklearn.metrics.roc_curve.
    """
    y_true = np.asarray(y_true)
    thresholds = np.linspace(1, 0, n_thresholds)

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        TP, TN, FP, FN = confusion_matrix_scratch(y_true, y_pred_t)

        tpr = TP / (TP + FN + 1e-15)        # = Recall
        fpr = FP / (FP + TN + 1e-15)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_scratch(y_true, y_proba, n_thresholds=100):
    """
    Tính diện tích dưới đường cong ROC bằng quy tắc hình thang (Trapezoidal Rule),
    thay vì gọi sklearn.metrics.roc_auc_score.
    """
    fpr, tpr, _ = roc_curve_scratch(y_true, y_proba, n_thresholds)
    # np.trapz tính diện tích hình thang; cần sắp xếp fpr tăng dần trước khi tích phân
    order = np.argsort(fpr)
    auc = np.trapz(tpr[order], fpr[order])
    return auc
```

**Giải thích ý tưởng**: mỗi khi hạ ngưỡng quyết định `t` xuống một chút, mô hình sẽ "dễ dãi" hơn khi gán nhãn Positive → TPR (Recall) và FPR đều tăng dần. Vẽ toàn bộ các cặp (FPR, TPR) này ra đồ thị chính là đường cong ROC ở mục 3.1; diện tích bên dưới nó chính là ROC-AUC.

### 4.4. Đối chiếu với `sklearn.metrics` để kiểm chứng

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Giả sử đã có y_test (nhãn thật) và y_pred, y_proba (từ model.predict / predict_proba)
metrics_scratch = classification_metrics_scratch(y_test, y_pred)
auc_scratch = roc_auc_scratch(y_test, y_proba)

print("=== SO SÁNH SCRATCH vs SKLEARN ===")
print(f"Accuracy  -> Scratch: {metrics_scratch['accuracy']:.4f} | Sklearn: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision -> Scratch: {metrics_scratch['precision']:.4f} | Sklearn: {precision_score(y_test, y_pred):.4f}")
print(f"Recall    -> Scratch: {metrics_scratch['recall']:.4f} | Sklearn: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  -> Scratch: {metrics_scratch['f1_score']:.4f} | Sklearn: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   -> Scratch: {auc_scratch:.4f} | Sklearn: {roc_auc_score(y_test, y_proba):.4f}")
```

Kỳ vọng: các giá trị Scratch và Sklearn sẽ khớp nhau (sai số rất nhỏ do làm tròn số và số lượng `n_thresholds` hữu hạn khi tính AUC bằng tay). Sau khi xác nhận công thức đã đúng, phần mục 5 dưới đây sẽ dùng thẳng `sklearn.metrics` cho công việc thực tế vì nó tối ưu và xử lý được nhiều trường hợp biên (edge case) hơn bản tự cài đặt.

---

## 5. Standard Evaluation Routine (Code Python)

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

## 6. Official References

Scikit-Learn Metrics Reference:

https://scikit-learn.org/stable/modules/model_evaluation.html
