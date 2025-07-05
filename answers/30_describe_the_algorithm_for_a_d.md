# Question 30: Describe the algorithm for a decision tree construction.

**Status:** ✅ Completed | **Words:** 139 | **Generated:** 2025-07-04T00:14:00Z

---

## Answer

Decision tree induction builds a predictive model by recursively partitioning the training data. Starting at the root, the algorithm evaluates all potential splits on every predictor and chooses the one that maximises impurity reduction—measured by information gain, Gini decrease or variance drop, depending on the task. The data are then divided according to this split, creating child nodes that are purer with respect to the target variable. The procedure repeats independently on each child until a stopping rule is met: minimum node size, maximum depth, or zero impurity. To combat over-fitting, post-pruning or cost-complexity pruning removes branches that do not improve validation error. Popular variants include ID3, C4.5 and CART; though they differ in split metrics and pruning heuristics, all follow this greedy top-down search, producing an interpretable set of if-then rules.

---
