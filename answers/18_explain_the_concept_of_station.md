# Question 18: Explain the concept of "stationarity" in time series analysis.

**Status:** ✅ Completed | **Words:** 149 | **Generated:** 2024-12-07T13:44:00Z

---

## Answer

A time series is stationary when its statistical properties remain constant under time shifts: the mean and variance are time-invariant and the autocovariance depends only on the lag, not on the calendar date. This stability ensures that historical patterns are informative about the future, allowing estimators and forecasts to converge. Econometric methods typically assume the weaker covariance-stationarity (constant first two moments) rather than strict distributional stationarity. Non-stationary series exhibit trends, seasonality, structural breaks, or unit roots, which can cause spurious regression results and exploding forecast errors. Visual plots, autocorrelation functions, and tests such as Augmented Dickey–Fuller, Phillips–Perron, or KPSS help detect non-stationarity. Remedies include differencing, logarithmic transformation, seasonal adjustment, or detrending to achieve an integrated series of order zero I(0). Once stationarity is restored, models like ARMA or state-space filters provide reliable inference and prediction.

---
