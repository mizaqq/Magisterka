# Question 31: Name at least three accuracy measures used in the classification quality assessment.

**Status:** ✅ Completed | **Words:** 144 | **Generated:** 2025-07-04T00:15:00Z

---

## Answer

Classification performance is typically summarised by several complementary metrics. 1) Accuracy—the share of correctly predicted labels among all observations—offers an intuitive overall rate but can be misleading in imbalanced data. 2) Precision—the proportion of predicted positives that are truly positive—captures false-alarm risk, crucial in spam or fraud screening. 3) Recall (or sensitivity)—the proportion of actual positives that are retrieved—emphasises missed detections, important in medical diagnostics. Combining these, the F1-score is the harmonic mean that balances precision and recall. Area Under the ROC Curve (AUC-ROC) evaluates ranking quality across all thresholds, while Matthews Correlation Coefficient provides a balanced single number even with severe skew. Selecting metrics therefore depends on class imbalance, decision cost and thresholding strategy.

---
