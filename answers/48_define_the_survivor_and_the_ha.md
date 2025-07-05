# Question 48: Define the survivor and the hazard functions.

**Status:** ✅ Completed | **Words:** 149 | **Generated:** 2025-07-04T00:32:00Z

---

## Answer

In time-to-event analysis, the survivor function S(t) gives the probability that a subject's event time T exceeds time t: S(t)=P(T>t). It starts at 1 and declines monotonically, providing an intuitive share of units still 'alive' at each horizon. The hazard function h(t) describes the instantaneous event rate among those who have survived just up to t; formally h(t)=f(t)/S(t) where f(t) is the density of T. It can also be expressed as the limit of P(t≤T<t+Δt | T≥t) divided by Δt as Δt→0. While S(t) is cumulative and bounded between 0 and 1, h(t) can exceed 1 and reveals how risk evolves over time—constant in an exponential model, increasing in ageing processes, or bathtub-shaped in reliability studies. Together they fully characterise the distribution of survival times since f(t)=−dS(t)/dt and S(t)=exp(−∫₀^t h(u)du).

---
