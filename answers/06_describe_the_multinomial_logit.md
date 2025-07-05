# Question 6: Describe the Multinomial Logit Model: its basic model, the methods of estimation of the model parameters and the interpretation of these parameters.

**Status:** ✅ Completed | **Words:** 153 | **Generated:** 2024-12-07T12:55:00Z

---

## Answer

The Multinomial Logit (MNL) model extends binary logit to choices among multiple mutually exclusive alternatives. It assumes that each decision maker associates utility U_{ij}=β_j'X_i+ε_{ij} with option j, where the error terms are i.i.d. Gumbel, producing the independence-of-irrelevant-alternatives property. Choice probabilities take the closed-form P_{ij}=exp(β_j'X_i)/Σ_k exp(β_k'X_i) with one alternative normalised as a reference for identification. Parameters are estimated by maximising the log-likelihood function using Newton–Raphson or quasi-Newton optimisation, and robust covariance matrices are obtained from the inverse Hessian. Coefficient signs indicate how covariates shift the log-odds of choosing an option, while marginal effects translate these impacts into absolute probability changes. Researchers often compute elasticities and willingness-to-pay measures for applied demand analysis. Despite computational simplicity, the restrictive IIA assumption motivates nested or mixed logit extensions when substitution patterns are complex.

---
