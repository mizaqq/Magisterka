# Question 23: Describe the elements of LP problem. Describe the methods that can be applied to solve it.

**Status:** ✅ Completed | **Words:** 205 | **Generated:** 2025-07-04T00:07:00Z

---

## Answer

A linear-programming problem consists of four core elements: decision variables, an objective function, linear constraints, and non-negativity conditions. The decision variables quantify the controllable choices—for example, production quantities—while the objective expresses a single performance metric, typically cost minimisation or profit maximisation, as a linear combination of these variables. Constraints translate technological, resource, or policy limits into a system of linear inequalities or equalities, and the non-negativity requirement enforces physical feasibility. Geometrically, the feasible region forms a convex polyhedron whose extreme points are candidate optima, guaranteeing global optimality at a vertex. For deterministic, medium-sized models the revised simplex algorithm remains the workhorse, exploiting sparsity and basis updates to traverse adjacent vertices efficiently. Large-scale network or block-angular structures benefit from specialised decompositions such as network simplex, Dantzig–Wolfe, or Benders. Interior-point methods solve the Karush-Kuhn-Tucker conditions directly and outperform simplex on very large, well-conditioned problems by converging in polynomial time. For integer or mixed-integer variants, branch-and-bound augmented by cutting planes and heuristics extends LP techniques to integrality. Open-source tools like COIN-OR and SCIP, and commercial solvers such as CPLEX or Gurobi, implement these algorithms with parallel processing and presolve routines that reduce problem size. Consequently, selecting a solution method hinges on problem structure, scale, and the need for exact versus approximate answers.

---
