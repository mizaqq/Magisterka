# Question 4: Describe the methods applied in modeling of the Binary Response Variables.

**Status:** ✅ Completed | **Words:** 178 | **Generated:** 2024-12-07T12:45:00Z

---

## Answer

Binary response variables represent outcomes with two possible states (success/failure, yes/no, employed/unemployed), requiring specialised econometric methods that account for the discrete nature of the dependent variable. The Linear Probability Model (LPM) applies ordinary least squares directly to binary outcomes, but suffers from heteroscedasticity and can produce predicted probabilities outside the [0,1] range. Logistic regression addresses these limitations by using the logit link function P(Y=1) = exp(β₀ + β₁X₁ + ... + βₚXₚ) / (1 + exp(β₀ + β₁X₁ + ... + βₚXₚ)), ensuring probabilities remain bounded. The probit model employs the standard normal cumulative distribution function as the link function, producing similar results to logit but with different tail behaviour. Maximum likelihood estimation (MLE) is used for both logit and probit models, with parameter interpretation requiring marginal effects calculation rather than direct coefficient analysis. Model selection criteria include the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC), while goodness-of-fit assessment uses pseudo-R² measures and classification accuracy metrics. These methods enable robust analysis of binary outcomes in economic and social science applications.

---
