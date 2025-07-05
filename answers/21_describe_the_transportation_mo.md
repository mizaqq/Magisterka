# Question 21: Describe the transportation model and its applications.

**Status:** ✅ Completed | **Words:** 223 | **Generated:** 2025-07-04T00:05:00Z

---

## Answer

The transportation model is a linear-programming formulation that determines the least-cost plan for shipping a homogeneous product from multiple supply sources to multiple demand destinations. It presumes known unit shipping costs, source capacities, and destination requirements, and seeks flows that satisfy all constraints while minimising the total cost objective. The coefficient matrix exhibits a special network structure—each variable appears in exactly one supply row and one demand column—so basic feasible solutions correspond to spanning trees with m + n − 1 positive arcs. Because of this structure, specialised algorithms such as the Northwest Corner, Vogel's Approximation, and MODI (u–v) method quickly identify an initial feasible solution and iterate to optimality far faster than generic simplex. Modern implementations use network simplex or cost-scaling algorithms running in near-linear time, enabling applications at global-supply-chain scale. Beyond freight logistics, the same formulation underpins workforce rostering, power-grid flow, and media-mix optimisation, where 'sources' and 'destinations' are abstract resources and demands. In public economics, it allocates vaccines from production sites to clinics, balancing cold-chain constraints and equity goals. Environmental planners apply it to minimise CO₂ in waste transport or to assign carbon credits across regions. The model's intuitive tableau and sensitivity ranges make it a staple case study in operations-research education. Overall, the transportation model translates diverse allocation problems into a solvable linear network, delivering cost transparency and operational efficiency.

---
