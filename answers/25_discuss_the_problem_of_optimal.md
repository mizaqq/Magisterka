# Question 25: Discuss the problem of optimal allocation in stratified sampling.

**Status:** ✅ Completed | **Words:** 190 | **Generated:** 2025-07-04T00:09:00Z

---

## Answer

Optimal allocation addresses how to distribute a fixed total sample size across strata to minimise the variance of an estimator or, equivalently, to achieve a target precision at minimum cost. Under equal per-unit sampling costs, the Neyman allocation rule assigns sample sizes in proportion to the product of stratum size N_h and standard deviation S_h, concentrating interviews where variability is greatest. When costs differ across strata, the generalised Neyman formula divides that product by the square root of the cost per unit, balancing variance reduction against interview expenses. Analysts often incorporate finite-population corrections and adjust for design effects to fine-tune these proportions in practice. In multi-variable surveys, compromise allocations—such as power allocation or optimisation routines—select stratum sizes that jointly minimise a weighted average of key variable variances. Departures from optimum, like proportional allocation or administrative quotas, typically inflate variance; sensitivity analysis quantifies this efficiency loss. Robust implementation also requires integer rounding and consideration of minimum stratum sizes to preserve estimator unbiasedness. Ultimately, optimal allocation formalises resource allocation in survey design, ensuring that each interview delivers the maximum possible information per unit cost.

---
