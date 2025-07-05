# Question 40: Identify and the discuss differences between the explanatory variables selection procedures for the regression and the classification models.

**Status:** ✅ Completed | **Words:** 187 | **Generated:** 2025-07-04T00:24:00Z

---

## Answer

Variable-selection goals overlap between regression and classification—simplify models, reduce over-fitting, and improve interpretability—but the criteria and algorithms differ because the loss functions do. In linear regression, stepwise AIC/BIC, adjusted R², and LASSO evaluate how predictors reduce continuous residual variance. Significance tests on coefficients (t-statistics, p-values) and multicollinearity diagnostics (VIF) guide inclusion. In classification, metrics are based on class separation: information gain, Gini impurity, mutual information, or area-under-ROC when wrappers evaluate subsets via cross-validated accuracy. Ensemble-based measures (permutation importance in random forests, SHAP) rank variables for nonlinear effects that regression may miss. Regularisers diverge too: logistic regression often uses L1 or elastic-net penalised likelihood, while tree ensembles rely on built-in split penalties. Furthermore, imbalance handling shapes selection in classification—SMOTE-augmented wrappers or cost-sensitive gains favour predictors that improve minority recall, whereas regression lacks this issue. Therefore, although generic subset-selection frameworks exist, effective practice tailors the criterion to the predictive objective: squared error for regression, misclassification risk or cross-entropy for classification.

---
