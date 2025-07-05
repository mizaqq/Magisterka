# Question 50: Describe the meaning of censored data.

**Status:** ✅ Completed | **Words:** 167 | **Generated:** 2025-07-04T00:34:00Z

---

## Answer

Censored data arise when an event of interest is only partially observed: we know it occurred after, before, or within an interval relative to the study timeline but not the exact time. The most common form is right-censoring, where subjects have not yet experienced the event by the study's end or are lost to follow-up, so their survival time is at least the observed duration. Left-censoring occurs when the event happened before observation began, giving only an upper bound, while interval-censoring states the event fell between two assessment visits. Censoring differs from missing data because the incomplete information still constrains the outcome distribution and, if handled properly, yields unbiased estimates. Survival models incorporate censoring via likelihood contributions of S(t) rather than f(t), ensuring that risk sets include only those still at risk. Ignoring censoring would overstate failure rates and distort covariate effects.

---
